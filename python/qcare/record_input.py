"""Annotated MIT-BIH input adapter, independent of prepared feature caches."""
from pathlib import Path
import json,numpy as np,wfdb
from .data import MAP
from .provenance import check_file

def load_record_data(root,record):
 root=Path(root);manifest=json.loads((root/'reports/dataset_manifest.json').read_text());entry=next(x for x in manifest['records'] if x['record']==record)
 for ext,info in entry['files'].items():check_file(root/'data'/(record+'.'+ext),info['sha256'])
 h=wfdb.rdheader(str(root/'data'/record));lead=h.sig_name.index('MLII') if 'MLII' in h.sig_name else 0;signal=wfdb.rdrecord(str(root/'data'/record),channels=[lead]).p_signal[:,0];ann=wfdb.rdann(str(root/'data'/record),'atr');peaks=[];labels=[];rrs=[];indices=[];history=[];prev=None;last=288
 for j,(v,symbol) in enumerate(zip(ann.sample,ann.symbol)):
  if symbol not in MAP:continue
  v=int(v);rr=v-prev if prev is not None else 288;median=float(np.median(history[-16:])) if history else 288.;std=float(np.std(history[-16:])) if len(history)>1 else 0.
  features=[rr/360,last/360,rr/max(median,1),rr/max(last,1),(rr-last)/max(median,1),std/max(median,1)]
  if v>=140 and v+219<len(signal):peaks.append(v);labels.append(MAP[symbol]);rrs.append(features);indices.append(j)
  prev=v;last=rr;history.append(rr)
 data={'record':np.full(len(peaks),record),'peak':np.array(peaks),'label':np.array(labels),'rr':np.array(rrs),'annotation_index':np.array(indices)}
 return signal,data
