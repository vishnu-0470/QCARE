from pathlib import Path
import json,hashlib,time
import numpy as np
import wfdb
from qcare.data import DS1,DS2,MAP,sha
from qcare.frontend import cwt_maps,quantize_map
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'experiments/cache';CACHE.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'reports/dataset_manifest.json').read_text())
if manifest['valid_records']!=44:raise ValueError('Complete verified 44-record dataset required')
validation=['108','114','209','223'];training=sorted(set(DS1)-set(validation))
records=[];peaks=[];labels=[];ann_indices=[];rrs=[];signals={};excluded={};counts={}
for rec in sorted(DS1+DS2):
 header=wfdb.rdheader(str(ROOT/'data'/rec));lead=header.sig_name.index('MLII') if 'MLII' in header.sig_name else 0
 signal=wfdb.rdrecord(str(ROOT/'data'/rec),channels=[lead]).p_signal[:,0]
 ann=wfdb.rdann(str(ROOT/'data'/rec),'atr');selected=[(i,int(p),MAP[s]) for i,(p,s) in enumerate(zip(ann.sample,ann.symbol)) if s in MAP]
 oldrr=[];prior_peak=None;last_rr=288;ignored=0
 for original,p,label in selected:
  rr=p-prior_peak if prior_peak is not None else 288
  median=float(np.median(oldrr[-16:])) if oldrr else 288.
  variation=float(np.std(oldrr[-16:])) if len(oldrr)>1 else 0.
  feature=[rr/360,last_rr/360,rr/max(median,1),rr/max(last_rr,1),(rr-last_rr)/max(median,1),variation/max(median,1)]
  if p>=140 and p+219<len(signal):
   records.append(rec);peaks.append(p);labels.append(label);ann_indices.append(original);rrs.append(feature)
  else:ignored+=1
  prior_peak=p;last_rr=rr;oldrr.append(rr)
 excluded[rec]=ignored;signals[rec]=signal;counts[rec]=np.bincount([y for _,p,y in selected if p>=140 and p+219<len(signal)],minlength=5).tolist()
records=np.array(records);peaks=np.array(peaks,dtype=np.int64);labels=np.array(labels,dtype=np.int64);N=len(labels)
np.savez(CACHE/'beats.npz',record=records,peak=peaks,label=labels,annotation_index=np.array(ann_indices),rr=np.array(rrs))
maps=np.lib.format.open_memmap(CACHE/'maps.npy',mode='w+',dtype=np.int8,shape=(N,1,60,60))
start=time.time()
for rec in sorted(signals):
 ix=np.flatnonzero(records==rec);signal=signals[rec]
 for base in range(0,len(ix),256):
  ids=ix[base:base+256];beats=np.stack([signal[p-140:p+220] for p in peaks[ids]])
  maps[ids]=quantize_map(cwt_maps(beats))
 print('CWT',rec,len(ix),'elapsed_s',round(time.time()-start,1),flush=True)
maps.flush()
rare=np.flatnonzero(np.isin(records,training)&np.isin(labels,[1,3,4]));aug_lookup=np.full(N,-1,dtype=np.int64);aug_lookup[rare]=np.arange(len(rare));np.save(CACHE/'aug_lookup.npy',aug_lookup)
aug=np.lib.format.open_memmap(CACHE/'augmented_maps.npy',mode='w+',dtype=np.int8,shape=(len(rare),41,1,60,60))
for starti in range(0,len(rare),8):
 ids=rare[starti:starti+8];beats=[]
 for i in ids:
  signal=signals[records[i]]
  for shift in range(-20,21):
   begin=int(peaks[i])-140+shift;lo=max(begin,0);hi=min(begin+360,len(signal));b=np.zeros(360);b[lo-begin:hi-begin]=signal[lo:hi];beats.append(b)
 augmented=quantize_map(cwt_maps(np.stack(beats))).reshape(len(ids),41,1,60,60);aug[starti:starti+len(ids)]=augmented
 if starti%160==0:print('Augmentation',starti,'/',len(rare),flush=True)
aug.flush()
report={'status':'PASS','protocol':'conventional_DS1_DS2','train_records':training,'validation_records':validation,'DS1':DS1,'DS2':DS2,'class_order':['N','S','V','F','Q'],'beat_window':{'pre':140,'post':219,'length':360,'edge_policy':'exclude incomplete base windows; zero-pad shifted training views if necessary'},'cwt':{'scales':list(range(1,61)),'omega':float(np.pi/2),'translation_stride':2,'pool':3,'map_shape':[1,60,60],'normalize':'per-beat max abs','quantization':'clamp(floor(128*M+0.5),-128,127)'},'total_beats':N,'class_counts_per_record':counts,'edge_exclusions_per_record':excluded,'rare_augmented_base_beats':len(rare),'augmentation_bias_inclusive':[40,80],'synthetic_ecg_training':False,'manifest_sha256':sha(ROOT/'reports/dataset_manifest.json'),'records_are_not_strictly_subject_disjoint':'201 (DS1) and 202 (DS2) are the same person','RR_feature_names':['previous_rr_s','preceding_rr_s','previous_rr_over_past16_median','previous_over_preceding_rr','rr_difference_over_past16_median','past16_std_over_median'],'RR_lookahead':'none: past R intervals only','cache_files':{f:sha(CACHE/f) for f in ['beats.npz','maps.npy','aug_lookup.npy','augmented_maps.npy']}}
(CACHE/'preparation.json').write_text(json.dumps(report,indent=2));(ROOT/'reports/dataset_preparation.json').write_text(json.dumps(report,indent=2));print('PASS prepared',N,'real annotated beats',flush=True)
