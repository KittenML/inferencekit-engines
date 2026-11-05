#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch the selected, hash-checked V19 sampler and V21 GDN implementation."""
import argparse,hashlib,json,os,re,sys,time
from pathlib import Path

def build_argv(root,arguments,executable):
    root=Path(root).resolve();recipe=json.loads((root/'selected_recipe.json').read_text());implementation=recipe['implementation']
    if (implementation['name'],implementation['driver'],implementation['gdn_mode'])!=('v19-sampler-v21-fixed-gdn','optimized_runtime.py','fixed-block'):
        raise ValueError('Unsupported selected implementation; no fallback is available')
    expected={'optimized_runtime.py','fixed_gdn.py','sampler_sort.py'}
    if set(implementation['files'])!=expected:raise ValueError('Incomplete implementation source identity')
    for name,wanted in implementation['files'].items():
        path=root/name
        if not re.fullmatch('[0-9a-f]{64}',wanted) or path.is_symlink() or not path.is_file() or path.resolve().parent!=root:
            raise ValueError('Invalid implementation source path or hash')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=wanted:raise ValueError('Selected implementation source differs: '+name)
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--implementation-receipt',default=str(root/'telemetry'/('implementation-'+str(os.getpid())+'-'+str(time.time_ns())+'.json')))
    options,forwarded=parser.parse_known_args(arguments)
    return [executable,str(root/'optimized_runtime.py'),'--package',str(root),'--receipt',str(Path(options.implementation_receipt).resolve()),'--gdn-mode','fixed-block','--','--telemetry',str(root/'telemetry/runtime.jsonl'),*forwarded]

def main():
    root=Path(__file__).resolve().parent;(root/'telemetry').mkdir(exist_ok=True)
    arguments=build_argv(root,sys.argv[1:],sys.executable);os.chdir(root);os.execv(sys.executable,arguments)
if __name__=='__main__':main()
