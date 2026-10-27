"""Official MIT-BIH records, exact current class map, auditable annotations."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,urllib.request,shutil,time
import numpy as np
import wfdb
DS1='101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230'.split()
DS2='100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234'.split()
CLASSES=['N','S','V','F','Q']
MAP={**dict.fromkeys('NLRej',0),**dict.fromkeys('aJASn',1),**dict.fromkeys('VEr',2),'F':3,**dict.fromkeys('/Qf',4)}
BASE='https://physionet.org/files/mitdb/1.0.0/'
MIRROR='https://physionet-open.s3.amazonaws.com/mitdb/1.0.0/'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sums(root,cached=None):
 p=Path(root)/'SHA256SUMS.txt'
 if not p.exists():
  if cached and (Path(cached)/p.name).exists():shutil.copy2(Path(cached)/p.name,p)
  else:p.write_bytes(urllib.request.urlopen(BASE+p.name,timeout=60).read())
 return {v[1].lstrip('*').removeprefix('./'):v[0] for line in p.read_text().splitlines() if len(v:=line.split())==2}
def inventory(root,destination,expected=None):
 root=Path(root); rows=[]
 for rec in sorted(DS1+DS2):
  row={'record':rec,'split':'DS1' if rec in DS1 else 'DS2','expected_files':[rec+'.'+x for x in ('hea','dat','atr')],'files':{},'status':'BLOCKED','exclusion_reason':None}
  for ext in ('hea','dat','atr'):
   p=root/(rec+'.'+ext);entry={'present':p.exists(),'bytes':p.stat().st_size if p.exists() else 0}
   if p.exists():entry['sha256']=sha(p);entry['publisher_hash_match']=None if expected is None else expected.get(p.name)==entry['sha256']
   row['files'][ext]=entry
  missing=[ext for ext,v in row['files'].items() if not v['present']]
  if missing:row['exclusion_reason']='Missing: '+','.join(missing)
  elif expected and any(not v.get('publisher_hash_match') for v in row['files'].values()):row['exclusion_reason']='Publisher hash mismatch (modified or incomplete source)'
  else:
   try:
    header=wfdb.rdheader(str(root/rec));ann=wfdb.rdann(str(root/rec),'atr')
    lead=header.sig_name.index('MLII') if 'MLII' in header.sig_name else 0
    signal=wfdb.rdrecord(str(root/rec),channels=[lead]).p_signal[:,0]
    if header.fs!=360 or header.n_sig!=2 or signal.size!=header.sig_len or not np.isfinite(signal).all():raise ValueError('Invalid sampling/channels/length/values')
    labels=np.array([MAP.get(s,-1) for s in ann.symbol]);keep=labels>=0
    if not keep.any():raise ValueError('No usable beat annotations')
    if np.any(ann.sample[keep]>=header.sig_len) or np.any(np.diff(ann.sample)<0):raise ValueError('Annotation locations outside record/nonmonotonic')
    row.update(status='PASS',sampling_hz=float(header.fs),channels=header.sig_name,selected_lead=header.sig_name[lead],samples=int(header.sig_len),annotation_count=len(ann.sample),beat_count=int(keep.sum()),class_counts=np.bincount(labels[keep],minlength=5).tolist(),ignored_nonbeat_annotations=int((~keep).sum()))
   except Exception as e:row['status']='FAIL';row['exclusion_reason']=type(e).__name__+': '+str(e)
  rows.append(row)
 report={'class_order':CLASSES,'required_records':44,'valid_records':sum(r['status']=='PASS' for r in rows),'records':rows,'protocol':'conventional_de_Chazal_DS1_DS2','same_subject_caveat':'201 in DS1 and 202 in DS2 originate from the same individual; conventional split is not strictly subject-disjoint','raw_synthetic_training':False}
 Path(destination).write_text(json.dumps(report,indent=2));return report

def download(root,backup,cache=None):
 root=Path(root);root.mkdir(exist_ok=True,parents=True);expected=sums(root,cache);backup=Path(backup)
 def fetch(name):
  p=root/name
  if p.exists() and sha(p)==expected[name]:return name,'already verified'
  if p.exists():
   b=backup/'data'/name;b.parent.mkdir(parents=True,exist_ok=True)
   if not b.exists():shutil.copy2(p,b)
  cp=Path(cache)/name if cache else None
  if cp is not None and cp.exists() and sha(cp)==expected[name]:data=cp.read_bytes();source='verified local official cache'
  else:
   err=None;data=None
   for prefix in (MIRROR,BASE,MIRROR):
    try:
     data=urllib.request.urlopen(prefix+name,timeout=60).read()
     if hashlib.sha256(data).hexdigest()!=expected[name]:raise ValueError('Publisher checksum mismatch')
     source=prefix+name;break
    except Exception as e:err=e;data=None
   if data is None:raise RuntimeError('Download failed: '+name) from err
  tmp=p.with_suffix(p.suffix+'.part');tmp.write_bytes(data);tmp.replace(p)
  return name,source
 log={};errors={}
 with ThreadPoolExecutor(max_workers=8) as pool:
  tasks={pool.submit(fetch,r+'.'+e):r+'.'+e for r in sorted(DS1+DS2) for e in ('hea','dat','atr')}
  for task,name in tasks.items():
   try:log[name]=task.result()
   except Exception as e:errors[name]=str(e)
 (root/'download_log.json').write_text(json.dumps({'files':log,'errors':errors,'checksum_source':BASE+'SHA256SUMS.txt'},indent=2))
 return expected,errors
