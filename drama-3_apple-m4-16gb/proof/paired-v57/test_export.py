import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import export_evidence as e
import validate_export as v

class Controls(unittest.TestCase):
 def test_actual_export_structure(self):self.assertEqual(v.inspect_data(P)['status'],'PASS_LOCAL_PUBLIC_EXPORT_STRUCTURE_AND_PRIVACY')
 def test_private_absolute_route_rejected(self):
  for part in ('Users','home','workspace','root','tmp'):
   with self.assertRaises(AssertionError):e.assert_public({'route':chr(47)+part+chr(47)+'fixture'})
 def test_endpoint_rejected(self):
  with self.assertRaises(AssertionError):e.assert_public({'endpoint':'http'+':'+chr(47)*2+'fixture'})
 def test_no_private_source_path_exported(self):
  for p in P.iterdir():
   if p.suffix in ('.json','.jsonl','.md'):self.assertFalse(e.forbidden_private(p.read_text()),p.name)
 def test_changed_pair_rejected(self):
  original=v.rows
  def changed(p):
   r=original(p)
   if Path(p).name=='PAIRS96.jsonl':r[0]['timing']['stock']['rtf']+=1
   return r
  with patch.object(v,'rows',side_effect=changed):
   with self.assertRaises(AssertionError):v.inspect_data(P)
 def test_duplicate_case_rejected(self):
  original=v.rows
  def changed(p):
   r=original(p)
   if Path(p).name=='CASES192.jsonl':r[-1]=r[0]
   return r
  with patch.object(v,'rows',side_effect=changed):
   with self.assertRaises(AssertionError):v.inspect_data(P)
 def test_missing_fresh_case_rejected(self):
  original=v.rows
  def changed(p):
   r=original(p)
   return r[:-1] if Path(p).name=='FRESH_PAIRS144.jsonl' else r
  with patch.object(v,'rows',side_effect=changed):
   with self.assertRaises(AssertionError):v.inspect_data(P)
 def test_changed_interval_rejected(self):
  original=v.load
  def changed(p):
   x=original(p)
   if Path(p).name=='COMPARATIVE_SPEEDUP.json':x['data']['speedup_ci95'][0]=1
   return x
  with patch.object(v,'load',side_effect=changed):
   with self.assertRaises(AssertionError):v.inspect_data(P)
 def test_no_old_native_or_original_grades_reused(self):
  x=e.load(P/'COMPARATIVE_SPEEDUP.json')['data']['fresh_quality_preservation']
  self.assertFalse(x['original_Tower6_grade_reuse'])
  self.assertEqual(x['native_actual144_each']['stock']['cases'],144)
 def test_failed_invocation_retained(self):
  x=e.load(P/'NATIVE_INVOCATION_PROVENANCE.json')['data']
  self.assertEqual(x['status'],'V1_FAILED_PATH_BEFORE_COMPARISON_V2_NATIVE_PASS')
  self.assertEqual(x['repair']['before_runner_sha256'],'ec0d22a26f0fb82df97b2785a564d191c1c2e3049fa946dafb840f9c9714a7a5')
 def test_original_receipts_separate_from_public_projection(self):
  source=e.load(P/'SOURCE_RECEIPTS.json')['receipts'];public=e.load(P/'COMPARATIVE_SPEEDUP.json')
  self.assertEqual(source['combined']['sha256'],public['original_receipt_sha256'])
  self.assertNotEqual(e.sha(P/'COMPARATIVE_SPEEDUP.json'),public['original_receipt_sha256'])
 def test_baseline_failure_and_tie_retained(self):
  x=e.load(P/'TESTED_BASELINE_SCOPE.json')['data']
  self.assertEqual(x['supported_setting_admission']['failed'],'recommended')
  self.assertEqual(x['selected'],'wire12b')
  self.assertEqual(x['all_pairwise_comparisons']['anchor11__wire12b']['decision']['verdict'],'inconclusive')
 def test_no_model_checker_or_ml_import_claim(self):
  self.assertEqual(e.load(P/'PUBLIC_REPRODUCTION_STATUS.json')['status'],'NOT_RUN_BY_THIS_EXPORT')
  self.assertFalse(any(x in sys.modules for x in ('torch','numpy','mlx','diffusers','PIL')))

if __name__=='__main__':unittest.main(verbosity=2)
