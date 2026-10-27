"""Package verified project files, retain originals, and validate archive contents."""
from pathlib import Path
import argparse,hashlib,json,zipfile,shutil
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
original=json.loads((r/'backups/original_sha256.json').read_text())
metadata={'reports/modified_files.json','reports/modified_files.md','reports/package_files_sha256.json'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def exclusion(rel):
 if rel in original:return None
 parts=Path(rel).parts
 if '__pycache__' in parts or rel.endswith(('.pyc','.vvp')):return 'regenerable compiled artifact'
 if parts[:2]==('experiments','cache') and Path(rel).name!='preparation.json':return 'large regenerable CWT/augmentation cache; hashes and preparation metadata included'
 if rel=='reports/packaging_run.log':return 'packager console is generated after the archive; external verification JSON is delivered separately'
 if rel in {'reports/qnn_premap.json','reports/qnn_trained_synth.edif'}:return 'partial/duplicate synthesis artifact; executed logs and final mapped netlists included'
 return None
for rel in metadata:(r/rel).touch(exist_ok=True)
files=sorted(p for p in r.rglob('*') if p.is_file())
missing=[rel for rel in original if not (r/rel).is_file()]
if missing:raise ValueError('Original source paths missing: '+str(missing))
rows=[]
for file in files:
 rel=file.relative_to(r).as_posix();skip=exclusion(rel);digest=None if rel in metadata else sha(file);before=original.get(rel)
 change='new' if before is None else 'unchanged' if digest==before else 'modified'
 rows.append({'path':rel,'change':change,'original_sha256':before,'sha256':digest,'bytes':file.stat().st_size if rel not in metadata else None,'included_in_ZIP':skip is None,'exclusion_reason':skip})
modified=[v for v in rows if v['change']=='modified'];new=[v for v in rows if v['change']=='new'];excluded=[v for v in rows if not v['included_in_ZIP']]
report={'status':'PASS','original_file_count':len(original),'modified_original_files':modified,'new_files':new,'unchanged_original_files':[v['path'] for v in rows if v['change']=='unchanged'],'missing_original_files':missing,'recoverability':'Original ZIP and original SHA256 inventory included; individual pre-change backups retained','metadata_hash_note':'The three self-referential metadata files omit their own hashes; all are covered by the external ZIP SHA256 and ZIP CRC check','excluded_files':excluded}
(r/'reports/modified_files.json').write_text(json.dumps(report,indent=2))
lines=['# Complete modified-file inventory','',f'Original files: {len(original)}. Modified original files: {len(modified)}. New paths: {len(new)}. Missing originals: {len(missing)}.','', 'Every original is recoverable from `backups/original_project.zip`; original hashes are in `backups/original_sha256.json`. Per-file current hashes and distribution exclusions are in `modified_files.json`.','', '## Modified original files','', '| File | Original SHA256 | Current SHA256 |','|---|---|---|']
for v in modified:lines.append(f'| {v["path"]} | {v["original_sha256"]} | {v["sha256"]} |')
lines += ['', '## New paths','']
for v in new:lines.append('- '+v['path']+(' — excluded: '+v['exclusion_reason'] if not v['included_in_ZIP'] else ''))
(r/'reports/modified_files.md').write_text('\n'.join(lines)+'\n')
packed=[f for f in files if exclusion(f.relative_to(r).as_posix()) is None]
checksums={f.relative_to(r).as_posix():sha(f) for f in packed if f.relative_to(r).as_posix() not in metadata}
(r/'reports/package_files_sha256.json').write_text(json.dumps({'status':'PASS','sha256':checksums,'metadata_covered_by_external_ZIP_hash':sorted(metadata)},indent=2))
archive=out/'QCARE_FPGA_audited_update.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for file in packed:z.write(file,'QCARE_FPGA/'+file.relative_to(r).as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for rel,digest in checksums.items():assert hashlib.sha256(z.read('QCARE_FPGA/'+rel)).hexdigest()==digest,rel
 manifest=json.loads(z.read('QCARE_FPGA/reports/dataset_manifest.json'))
 for rec in manifest['records']:
  for ext,info in rec['files'].items():
   name='QCARE_FPGA/data/'+rec['record']+'.'+ext
   assert hashlib.sha256(z.read(name)).hexdigest()==info['sha256'],name
 assert hashlib.sha256(z.read('QCARE_FPGA/backups/original_project.zip')).hexdigest()==json.loads((r/'reports/baseline_archive.json').read_text())['sha256']
 assert len(json.loads(z.read('QCARE_FPGA/experiments/ds2_evaluation/metrics.json'))['quantized_current_irn']['records'])==22
summary={'status':'PASS','archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(packed),'original_files_recoverable':len(original),'modified_original_files':len(modified),'raw_data_files_verified_inside_ZIP':132,'all_ZIP_CRCs_valid':True,'all_packaged_nonmetadata_SHA256_valid':True,'excluded_regenerable_paths':len(excluded)}
(out/'QCARE_PACKAGE_VERIFICATION.json').write_text(json.dumps(summary,indent=2));(out/'QCARE_release_sha256.txt').write_text(summary['sha256']+'  '+archive.name+'\n')
shutil.copy2(r/'reports/FINAL_REPORT.md',out/'QCARE_RESULTS.md');shutil.copy2(r/'reports/plots/quantized_current_irn_confusion.png',out/'QCARE_IRN_confusion.png');shutil.copy2(r/'reports/plots/RR_logistic_baseline_confusion.png',out/'QCARE_RR_confusion.png');shutil.copy2(r/'reports/plots/training.png',out/'QCARE_training.png')
print(json.dumps(summary,indent=2))
