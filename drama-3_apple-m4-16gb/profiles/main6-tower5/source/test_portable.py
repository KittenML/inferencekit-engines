"""Small stdlib controls only; no ML imports, model data or numerical proof."""
import ast,copy,datetime,errno,hashlib,importlib.util,io,json,os,stat,struct,sys,tempfile,time,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tower5_builder as b
import portable_common as c
import profile_setup

P=Path(__file__).resolve().parent


def tiny_sft(root):
    data=struct.pack('<HH',0x3f80,0x4000)
    entry={'shape':[1,2],'dtype':'BF16','data_offsets':[0,4]}
    header=json.dumps({'tensor':entry},separators=(',',':')).encode()
    path=root/'tiny.safetensors';path.write_bytes(struct.pack('<Q',len(header))+header+data)
    spec={'file':str(b.ORIGINAL_PREFIX/'tiny.safetensors'),'key':'tensor','shape':[1,2],'dtype':'BF16','data_offsets':[0,4],
          'file_bytes':path.stat().st_size,'file_sha256':c.sha(path),'header_bytes':len(header),'header_sha256':hashlib.sha256(header).hexdigest()}
    return path,spec,data


class Controls(unittest.TestCase):
    def test_retained_runtime_bytes_and_modes(self):self.assertEqual(len(c.verify_measured()),36)
    def test_all78_234_registered_order(self):
        plan=c.load(P/'TENSOR_PLAN_V52_V1.json');manifest=c.load(P/'model-metadata/TOWER5_OVERLAY_MANIFEST_V1.json')
        self.assertEqual(c.sha(P/'TENSOR_PLAN_V52_V1.json'),b.PLAN_SHA256)
        self.assertEqual(len(plan['targets']),78)
        self.assertEqual(list(manifest['tensors']),[r['module']+'.'+s for r in plan['targets'] for s in ('weight','scales','biases')])
    def test_all_npy_headers_exact_measured_function(self):
        tree=ast.parse((P/'MEASURED_BUILDER_V52.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='npy_header')
        scope={'struct':struct};exec(compile(ast.Module(body=[node],type_ignores=[]),'<measured header>','exec'),scope)
        for item in c.load(P/'model-metadata/TOWER5_OVERLAY_MANIFEST_V1.json')['tensors'].values():
            header=b.npy_header(item['stored_dtype'],item['shape'])
            self.assertEqual(header,scope['npy_header'](item['stored_dtype'],item['shape']))
            self.assertEqual(len(header)%64,0)
            self.assertEqual(len(header)+__import__('math').prod(item['shape'])*(4 if item['stored_dtype']=='<u4' else 2),item['npy_bytes'])
    def test_stored_zip_open_deterministic(self):
        result=[]
        for _ in range(2):
            f=io.BytesIO()
            with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_STORED) as archive:
                with archive.open('fixture.npy','w') as entry:entry.write(b.npy_header('<u4',(1,))+struct.pack('<I',7))
            result.append(f.getvalue())
        self.assertEqual(*result)
    def test_independent_npz_reread(self):
        raw=struct.pack('<I',7);header=b.npy_header('<u4',(1,));data=header+raw
        tensors={'fixture':{'stored_dtype':'<u4','shape':[1],'npy_bytes':len(data),
                           'npy_sha256':hashlib.sha256(data).hexdigest(),'raw_le_sha256':hashlib.sha256(raw).hexdigest()}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tiny.npz'
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
                with archive.open('fixture.npy','w') as entry:entry.write(data)
            self.assertEqual(b.audit_npz(path,tensors,lambda:None)['members'],1)
            wrong=copy.deepcopy(tensors);wrong['fixture']['raw_le_sha256']='a'*64
            with self.assertRaises(ValueError):b.audit_npz(path,wrong,lambda:None)
    def test_npz_unregistered_member_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tiny.npz'
            with zipfile.ZipFile(path,'w') as archive:
                with archive.open('extra.npy','w') as entry:entry.write(b'fixture')
            with self.assertRaises(ValueError):b.audit_npz(path,{},lambda:None)
    def test_q5_eval_precedes_q6_constructor(self):
        tree=ast.parse((P/'tower5_builder.py').read_text());events=[]
        for node in sorted((n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='mx'),key=lambda n:n.lineno):
            if node.func.attr=='quantize':events.append(('quantize',next(k.value.value for k in node.keywords if k.arg=='bits')))
            elif node.func.attr=='eval' and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id in ('q5','q6'):events.append(('eval',node.args[0].id))
        self.assertEqual(events[:4],[('quantize',5),('eval','q5'),('quantize',6),('eval','q6')])
    def test_no_numerical_module_imports(self):
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('mlx.core',sys.modules);self.assertNotIn('torch',sys.modules)
    def test_guard_control_cannot_be_artifact_success(self):
        worker=(P/'builder_worker.py').read_text();supervisor=(P/'build_guarded.py').read_text()
        self.assertLess(worker.index('if a.guard_control:'),worker.index('import mlx.core'))
        self.assertIn('PASS_GUARD_CONTROL_NOT_NUMERICAL',worker)
        self.assertIn("actual['model_revision'] is None and not output.exists()",supervisor)
    def test_origin_and_derivation_notice_retained(self):
        self.assertEqual(c.sha(P/'model-metadata/origin.json'),b.TARGET_ORIGIN_SHA256)
        notice=c.load(P/'model-metadata/TOWER5_OVERLAY_NOTICE_V1.json')
        self.assertEqual(notice['builder_sha256'],'2c34ddcdc627b3d0438fe4a03498d82c2f444f5973bfaddbc4680bb1405a9a9d')
        self.assertEqual(c.load(P/'model-metadata/origin.json')['files']['tower5-overlay.npz']['sha256'],'7b67984c3fa7e4a028186e16b00b3f27a132239daf2778707d7642f16dafe1b2')
    def test_explicit_prefix_relocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'a').write_bytes(b'x')
            self.assertEqual(b.relocated(root,str(b.BASE_PREFIX/'a'),b.BASE_PREFIX),root/'a')
    def test_reject_prefix_escape_and_nested(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for text in (str(b.BASE_PREFIX/'../a'),str(b.BASE_PREFIX/'nested/a'),'unregistered/a'):
                with self.assertRaises(ValueError):b.relocated(root,text,b.BASE_PREFIX)
    def test_reject_member_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder).resolve();(p/'real').write_bytes(b'x');(p/'alias').symlink_to(p/'real')
            with self.assertRaises(ValueError):b.regular_member(p,'alias')
    def test_reject_absolute_flat_member(self):
        with self.assertRaises(ValueError):b.regular_member(P,'/absolute')
    def test_tiny_header_and_raw_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();_,spec,data=tiny_sft(root)
            self.assertEqual(b.read_tensor(root,b.ORIGINAL_PREFIX,spec,lambda:None),data)
    def test_changed_header_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();_,spec,_=tiny_sft(root);spec['header_sha256']='a'*64
            with self.assertRaises(ValueError):b.read_tensor(root,b.ORIGINAL_PREFIX,spec,lambda:None)
    def test_truncated_source_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path,spec,_=tiny_sft(root);path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):b.read_tensor(root,b.ORIGINAL_PREFIX,spec,lambda:None)
    def test_hash_change_during_read_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'file';path.write_bytes(b'abc')
            def mutate():path.write_bytes(b'changed-size')
            with self.assertRaises(ValueError):b.digest(path,mutate)
    def test_hardlink_never_opens_shared_for_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';dest=root/'dest';source.write_bytes(b'abc');mode=b.share_file(source,dest,{'bytes':3,'sha256':c.sha(source)},lambda:None)
            self.assertEqual(mode,'hardlink');self.assertEqual(source.stat().st_ino,dest.stat().st_ino);self.assertEqual(source.read_bytes(),b'abc')
    def test_cross_device_copy_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';dest=root/'dest';source.write_bytes(b'abc')
            with patch.object(b.os,'link',side_effect=OSError(errno.EXDEV,'fixture')):mode=b.share_file(source,dest,{'bytes':3,'sha256':c.sha(source)},lambda:None)
            self.assertEqual(mode,'copy');self.assertNotEqual(source.stat().st_ino,dest.stat().st_ino)
    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);src=p/'src';dst=p/'dst';src.write_bytes(b'abc');dst.write_bytes(b'protected')
            with self.assertRaises(ValueError):b.share_file(src,dst,{'bytes':3,'sha256':c.sha(src)},lambda:None)
            self.assertEqual(dst.read_bytes(),b'protected')
    def test_wrong_copy_hash_preserves_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);src=p/'src';dst=p/'dst';src.write_bytes(b'abc')
            with self.assertRaises(ValueError):b.share_file(src,dst,{'bytes':3,'sha256':'0'*64},lambda:None)
            self.assertTrue(dst.exists())
    def test_deadline_finite_and_future(self):
        self.assertIsNotNone(c.deadline(seconds=30))
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):c.deadline(seconds=value)
    def test_deadline_no_stale_or_naive_date(self):
        for value in ('2000-01-01T00:00:00+00:00','2030-01-01T00:00:00'):
            with self.assertRaises(ValueError):c.deadline(utc_value=value)
    def test_python_venv_symlink_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);link=p/'venv-python';link.symlink_to(sys.executable)
            self.assertEqual(c.selected_python(str(link)),link)
            self.assertNotEqual(c.selected_python(str(link)),link.resolve())
    def test_explicit_root_no_relative_or_traversal(self):
        for value in ('relative','/tmp/a/../b'):
            with self.assertRaises(ValueError):c.explicit(value)
    def test_output_cannot_be_in_protected_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(ValueError):c.separate(root/'new',root)
    def test_pressure_heartbeat_boundary(self):
        ready={'preflight':{'observations':{'essential_finished_monotonic_ns':10**9}}}
        with tempfile.TemporaryDirectory() as folder,patch.object(c.time,'monotonic_ns',return_value=6*10**9):c.pressure_fresh(Path(folder)/'missing',ready)
        with tempfile.TemporaryDirectory() as folder,patch.object(c.time,'monotonic_ns',return_value=6*10**9+1):
            with self.assertRaises(ValueError):c.pressure_fresh(Path(folder)/'missing',ready)
    def test_triggered_pressure_rejected(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(c.time,'monotonic_ns',return_value=1):
            p=Path(folder)/'log';p.write_text(json.dumps({'sample_finished_monotonic_ns':1,'pressure_guard':{'triggered':True}})+'\n')
            with self.assertRaises(ValueError):c.pressure_fresh(p,{'preflight':{'observations':{'essential_finished_monotonic_ns':1}}})
    def test_guard_policy_source_byte_exact(self):
        self.assertEqual(c.sha(P/'guard-source/memory_guard.py'),'f7bb06071c8dd749ecb9b5c52bbdef48247dc62b69189261313187c50dfff386')
        self.assertEqual(c.sha(P/'guard-source/memory_watch.py'),'d0e3ba6c34b76534dc1e068316ede8c424922d49ae4945dc9f1783821a811be6')
        self.assertEqual(c.sha(P/'guard-source/owned_observation.py'),'244991fe1846e898c9c1cfecd9cfeff703278ab63eceed198ab03bd7676dd0dd')
    def test_cleanup_source_byte_exact(self):self.assertEqual(c.sha(P/'guard-source/cleanup_owned.py'),'e60e87b299948296afb1f1505a74aafa9675cff0d74d633ecbd5c5393989bb91')
    def test_no_hardcoded_private_deadline_active_sources(self):
        for name in ('tower5_builder.py','build_guarded.py','builder_worker.py','portable_deadline_watch.py','profile_setup.py'):
            text=(P/name).read_text();self.assertNotIn('2026-10-26T08',text);self.assertNotIn('prior_owner_manifests',text);self.assertNotIn('CLOSED_V51',text)
    def test_parent_installer_selects742_not_overlay_flags(self):
        parent=c.load(P/'parent-installer-source/selected_recipe.json')
        self.assertEqual(parent['revision'],b.BASE_REVISION)
        self.assertNotIn('--source-original-BF16',parent['conversion_flags'])
        self.assertNotIn('--tower-group-size',parent['conversion_flags'])
        self.assertEqual(c.sha(P/'parent-installer-source/sources.lock.json'),b.PARENT_LOCK_SHA256)
    def test_fresh_setup_never_calls_measured_installer(self):
        text=(P/'profile_setup.py').read_text()
        self.assertIn("out/'parent-work/install.sh'",text);self.assertNotIn('subprocess',text)
    def test_builder_origin_atomic_completion_marker(self):
        text=(P/'tower5_builder.py').read_text();self.assertIn('origin.partial',text)
        self.assertLess(text.index('MLX limit restoration failed'),text.index('(output / "origin.partial")'))


if __name__=='__main__':unittest.main(verbosity=2)
