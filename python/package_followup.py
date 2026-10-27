"""Create an updated project ZIP while preserving the earlier delivered archive."""
from pathlib import Path
import argparse,tempfile,subprocess,sys,shutil,json,hashlib
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir).resolve();out.mkdir(exist_ok=True,parents=True)
with tempfile.TemporaryDirectory(prefix='qcare_followup_package_',dir=r.parent) as tmp:
 c=subprocess.run([sys.executable,str(r/'python/package_release.py'),'--output-dir',tmp],cwd=r,capture_output=True,text=True)
 if c.returncode:raise RuntimeError(c.stdout+c.stderr)
 tmp=Path(tmp);archive=out/'QCARE_FPGA_accuracy_update.zip';shutil.copy2(tmp/'QCARE_FPGA_audited_update.zip',archive);v=json.loads((tmp/'QCARE_PACKAGE_VERIFICATION.json').read_text());v['archive']=archive.name;v['results']=json.loads((r/'reports/LATEST_RESULTS.json').read_text());(out/'QCARE_LATEST_PACKAGE_VERIFICATION.json').write_text(json.dumps(v,indent=2));(out/'QCARE_latest_sha256.txt').write_text(v['sha256']+'  '+archive.name+'\n')
shutil.copy2(r/'RUN_GUIDE.md',out/'QCARE_RUN_GUIDE.md');shutil.copy2(r/'reports/LATEST_REPORT.md',out/'QCARE_LATEST_RESULTS.md')
for name in ['fusion90_confusion.png','software93_confusion.png']:shutil.copy2(r/'reports/plots/accuracy_followup'/name,out/name)
print(json.dumps({k:v[k] for k in ['status','archive','bytes','sha256','files','raw_data_files_verified_inside_ZIP']},indent=2))
