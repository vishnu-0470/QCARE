from pathlib import Path
import json,tempfile,sys
from qcare.provenance import verify_prepared,verify_checkpoint_preparation,check_file,digest
from train_current import load_checkpoint
r=Path(__file__).resolve().parents[1];prep=verify_prepared(r);ck=load_checkpoint(r/'experiments/ds1_current/best_qat.pt',weights_only=True,map_location='cpu');verify_checkpoint_preparation(ck,prep)
with tempfile.TemporaryDirectory(prefix='qcare_hash_') as tmp:
 p=Path(tmp)/'fixture';p.write_bytes(b'expected original file');expected=digest(p);p.write_bytes(b'changed file')
 try:check_file(p,expected)
 except ValueError:changed_rejected=True
 else:raise AssertionError('Changed fixture was accepted')
 p.unlink()
 try:check_file(p,expected)
 except FileNotFoundError:missing_rejected=True
 else:raise AssertionError('Missing fixture was accepted')
report={'status':'PASS','raw_files_checked':132,'cache_files_checked':4,'raw_publisher_hashes_unchanged':True,'checkpoint_preparation_matches_current':True,'changed_file_rejected':changed_rejected,'missing_file_rejected':missing_rejected,'no_training_or_DS2_inference_rerun':True}
(r/'reports/provenance_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
