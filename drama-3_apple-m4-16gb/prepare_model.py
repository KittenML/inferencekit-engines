#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the selected pinned conversion and verify its complete content manifest."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from download_model import file_hash
from serve import canonical

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-model',type=Path,help='Verify the complete locked artifact before linking an existing local cache; never modify it')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    recipe_path=root/'selected_recipe.json'
    if not recipe_path.exists():raise SystemExit('No validated default recipe has been selected in this private draft.')
    recipe=json.loads(recipe_path.read_text());lock=json.loads((root/'sources.lock.json').read_text());track=recipe['track'];spec=lock['models'][track]
    output=root/'models'/track
    output.parent.mkdir(parents=True,exist_ok=True)
    if args.reuse_model:
        reused=args.reuse_model.resolve(strict=True)
        if not reused.is_dir():raise SystemExit('Reuse path must be a model directory')
        if output.exists() or output.is_symlink():raise SystemExit('Model target already exists; refusing to replace it')
        candidate=reused
    else:candidate=output
    if not args.reuse_model and not (output/'origin.json').exists():
        if output.exists() and any(output.iterdir()):raise SystemExit('Unregistered model directory exists; refusing to overwrite user files.')
        original=root/'models/original-fp8';codec=root/'models/upstream-codec';codec.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,str(root/'download_model.py'),'--track','original-fp8','--output',str(original),'--acknowledge-approved-host'],check=True)
        from huggingface_hub import hf_hub_download
        codec_spec=lock['models']['upstream-3bit'];item=next(v for v in codec_spec['files'] if v['path']=='codec.safetensors')
        codec_file=Path(hf_hub_download(codec_spec['id'],'codec.safetensors',revision=codec_spec['revision'],local_dir=codec))
        if codec_file.stat().st_size!=item['bytes'] or file_hash(codec_file)!=item['lfs_sha256']:raise RuntimeError('Codec hash mismatch')
        command=[sys.executable,str(root/'convert_low_memory.py'),'--source',str(original),'--output',str(output),'--codec-from',str(codec),*recipe['conversion_flags']]
        subprocess.run(command,check=True)
        # Preserve the original artifact's derivation notice; record this install's actual command separately.
        notice={'original_model':lock['original_model']['id'],'original_revision':lock['original_model']['revision'],'mlx_audio_commit':lock['mlx_audio_commit'],'conversion_command':spec['derivation']['conversion_command'],'quantization_label':spec['quant']}
        (output/'CONVERSION_NOTICE.txt').write_text('Built with Fish Audio. Local conversion; unofficial and not endorsed by Fish Audio.\n'+canonical(notice)+'\n')
        (root/'reproduction-command.json').write_text(json.dumps({'command':command,'reference_artifact_revision':spec['revision'],'notice_retains_original_creation_command':True},indent=2)+'\n')
    if {p.name for p in candidate.glob("*.safetensors")} != {item["path"] for item in spec["files"] if item["path"].endswith(".safetensors")}:raise RuntimeError("Unregistered checkpoint shards present or pinned shards missing")
    files={}
    for item in spec['files']:
        path=candidate/item['path']
        if not path.is_file() or path.stat().st_size!=item['bytes']:raise RuntimeError('Reproduced file size mismatch: '+item['path'])
        digest=file_hash(path)
        if item['lfs_sha256'] and digest!=item['lfs_sha256']:raise RuntimeError('Reproduced content differs from selected artifact: '+item['path'])
        if not item['lfs_sha256'] and file_hash(path,'sha1',git_blob=True)!=item['git_blob_sha1']:raise RuntimeError('Reproduced metadata differs: '+item['path'])
        files[item['path']]={'sha256':digest,'bytes':item['bytes']}
    origin={'model':spec['id'],'revision':spec['revision'],'files':files}
    if args.reuse_model:
        existing=json.loads((candidate/'origin.json').read_text())
        if any(existing.get(k)!=origin[k] for k in ('model','revision','files')):raise RuntimeError('Reused artifact origin does not match complete verified lock')
        output.symlink_to(candidate,target_is_directory=True)
    else:(output/'origin.json').write_text(json.dumps(origin,indent=2)+'\n')
    (root/'model-verification.json').write_text(json.dumps({'model':spec['id'],'revision':spec['revision'],'files':files,'reused_read_only':bool(args.reuse_model)},indent=2)+'\n')
    print(json.dumps({'model':spec['id'],'revision':spec['revision'],'files_verified':len(files),'reused_read_only':bool(args.reuse_model)}))

if __name__=='__main__':main()
