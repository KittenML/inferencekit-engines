#!/usr/bin/env python3
"""Bind the candidate overlay, source and resolved config before launching."""
import argparse, json
from pathlib import Path
from fp8_install import ROOT, PATCH, SCALE, sha, tree, wheel_inventory
from verify_install import verify
parser=argparse.ArgumentParser(); parser.add_argument('--configuration', required=True, type=Path); args=parser.parse_args()
r=ROOT/'runtime'; cfg=args.configuration.resolve(strict=True)
verify(r, ROOT/'model-artifacts.json', cfg)
config=json.loads(cfg.read_text()); record=json.loads((r/'fp8-install-complete.json').read_text())
manifest=json.loads((ROOT/'engine/adapter/SOURCE_MANIFEST.json').read_text())
if record.get('status')!='CPU_INSTALLED_NOT_RUN' or record.get('base_receipt_sha256')!=sha(r/'install-complete.json') or record.get('source_manifest_sha256')!=sha(ROOT/'engine/adapter/SOURCE_MANIFEST.json') or record.get('engine_manifest_sha256')!=sha(ROOT/'engine-manifest.json') or record.get('patch_sha256')!=PATCH or record.get('overlay_fp8_sha256')!=SCALE: raise SystemExit('candidate installation receipt differs')
if config.get('fp8_overlay_root')!=str((r/'fp8-overlay').resolve()) or config.get('native_source_root')!=str((r/'vllm-omni').resolve()) or config.get('candidate_patch_sha256')!=PATCH or config.get('fp8_scale_source_sha256')!=SCALE: raise SystemExit('candidate configuration differs')
if tree(r/'fp8-overlay')!=record.get('overlay_tree') or tree(r/'fp8-overlay')!=wheel_inventory(r, patched=True): raise SystemExit('overlay tree differs')
for row in manifest['files']:
 p=r/'vllm-omni'/row['path']
 if p.is_symlink() or sha(p)!=row['sha256']: raise SystemExit('candidate source differs: '+row['path'])
print('PASS: resolved model, candidate sources and isolated overlay; runtime execution UNPROVEN')
