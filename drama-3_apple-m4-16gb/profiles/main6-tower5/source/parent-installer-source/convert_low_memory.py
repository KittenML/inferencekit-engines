#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct original-FP8 conversion using the pinned upstream converter, smaller output shards."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--codec-from", required=True)
    p.add_argument("--expert-bits", type=int, default=2)
    p.add_argument("--expert-group-size", type=int, choices=(32,64,128), default=128)
    p.add_argument("--dense-bits", type=int, default=8)
    p.add_argument("--tower-bits", type=int, default=8)
    p.add_argument("--head-bits", type=int, default=8, help="Residual codebook heads; 0 preserves original BF16")
    p.add_argument("--semantic-head-bits",type=int,choices=(0,2,3,4,6,8),default=None,help="Main semantic/EOS head; default follows dense bits, 0 preserves BF16")
    p.add_argument("--expert-bits-map")
    p.add_argument("--expert-group-size-map", help="JSON layer-to-group-size map; unspecified layers retain expert-group-size")
    p.add_argument("--expert-quantizer",choices=("upstream","mse-clip-v1"),default="upstream")
    p.add_argument("--embedding-bits", type=int, choices=(0,4,8), default=0)
    args = p.parse_args()
    import mlx.core as mx
    from mlx_audio.tts.models.fish_drama3 import convert as upstream
    from download_model import file_hash
    lock = json.loads(Path(__file__).with_name("sources.lock.json").read_text())
    origin = json.loads((Path(args.source)/"origin.json").read_text())
    original = lock["original_model"]
    if origin["model"] != original["id"] or origin["revision"] != original["revision"]:
        raise RuntimeError("conversion source is not the pinned originalFP8")
    for item in original["files"]:
        file = Path(args.source)/item["path"]
        if file.stat().st_size != item["bytes"]:
            raise RuntimeError("source file size mismatch")
        if item["lfs_sha256"] and file_hash(file) != item["lfs_sha256"]:
            raise RuntimeError("source weight checksum mismatch")
        if not item["lfs_sha256"] and file_hash(file, "sha1", git_blob=True) != item["git_blob_sha1"]:
            raise RuntimeError("source metadata checksum mismatch")
    codec = Path(args.codec_from)/"codec.safetensors"
    expected_codec = next(x for x in lock["models"]["upstream-3bit"]["files"] if x["path"]==codec.name)
    if file_hash(codec) != expected_codec["lfs_sha256"]:
        raise RuntimeError("codec differs from pinned upstream conversion")
    # Folder-local shard scheduling and the optional MSE-clipping extension are ours.
    # Other conversion operations use the pinned upstream converter.
    mx.set_cache_limit(0)
    upstream.SHARD_BYTES = 512 * 1024 * 1024
    mse_stats=[]
    group_map = {}
    if args.expert_group_size_map:
        raw_map = json.loads(Path(args.expert_group_size_map).read_text())
        if not isinstance(raw_map, dict) or any(not str(k).isdigit() or not 0 <= int(k) < 40
                or type(v) is not int or v not in (32, 64, 128) for k, v in raw_map.items()):
            raise ValueError("expert group map requires layer0..39 and group32/64/128")
        group_map = {int(k):v for k,v in raw_map.items()}
    if args.embedding_bits or args.semantic_head_bits is not None or args.expert_quantizer!="upstream" or group_map:
        original_add = upstream.ShardWriter.add
        def add_with_embedding(self, name, value, bits=None, group_size=64):
            if name == "backbone.model.embed_tokens.weight" and args.embedding_bits:
                bits, group_size = args.embedding_bits, 64
            if name == "semantic_head.weight" and args.semantic_head_bits is not None:
                bits, group_size = args.semantic_head_bits, 64
            if ".mlp.switch_mlp." in name and group_map:
                layer = int(name.split(".layers.")[1].split(".")[0])
                group_size = group_map.get(layer, group_size)
            if args.expert_quantizer=="mse-clip-v1" and ".mlp.switch_mlp." in name and bits==2:
                from mse_quant import quantize_mse_clip_v1
                packed,scales,biases,stats=quantize_mse_clip_v1(value,group_size)
                path=name.removesuffix(".weight")
                items={name:packed,path+".scales":scales,path+".biases":biases}
                self.quantization[path]={"group_size":group_size,"bits":bits}
                mx.eval(list(items.values()));self.pending.update(items)
                self.pending_bytes+=sum(v.nbytes for v in items.values())
                if self.pending_bytes>=upstream.SHARD_BYTES:self.flush()
                stats["path"]=path;mse_stats.append(stats);print(json.dumps({"mse_quantization":stats}),flush=True)
                return
            return original_add(self, name, value, bits, group_size)
        upstream.ShardWriter.add = add_with_embedding
    bits_map = {int(k):v for k,v in json.loads(Path(args.expert_bits_map).read_text()).items()} if args.expert_bits_map else None
    upstream.convert(Path(args.source),Path(args.output),expert_bits=args.expert_bits,dense_bits=args.dense_bits,
                     tower_bits=args.tower_bits,codec=False,expert_group_size=args.expert_group_size,
                     head_bits=args.head_bits,expert_bits_map=bits_map)
    shutil.copy2(codec,Path(args.output)/codec.name)
    if mse_stats:(Path(args.output)/"mse-conversion-v1.json").write_text(json.dumps(mse_stats,indent=2)+"\n")
    print(json.dumps({"peak_conversion_mlx_bytes":mx.get_peak_memory(),"source_revision":original["revision"],
                      "expert_bits":args.expert_bits,"expert_group_size":args.expert_group_size,
                      "dense_bits":args.dense_bits,"tower_bits":args.tower_bits,"head_bits":args.head_bits,
                      "embedding_bits":args.embedding_bits,"semantic_head_bits":args.semantic_head_bits,"expert_quantizer":args.expert_quantizer,"expert_group_size_map":group_map,"output_shard_bytes":upstream.SHARD_BYTES}),flush=True)


if __name__ == "__main__":
    main()
