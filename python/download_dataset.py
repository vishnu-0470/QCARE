from pathlib import Path
import sys,json
from qcare.data import sums,inventory,download
root=Path(__file__).resolve().parents[1];cache=root.parent/'mitdb'
expected=sums(root/'data',cache)
inventory(root/'data',root/'reports/dataset_manifest_before.json',expected)
expected,errors=download(root/'data',root/'backups/pre_repair',cache)
m=inventory(root/'data',root/'reports/dataset_manifest.json',expected)
print(json.dumps({'valid_records':m['valid_records'],'required':44,'errors':errors},indent=2),flush=True)
if m['valid_records']!=44:sys.exit(1)
