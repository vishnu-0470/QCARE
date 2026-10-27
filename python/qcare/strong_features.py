"""Extended causal RR context and real beat morphology for software diagnostics."""
from pathlib import Path
import numpy as np,wfdb
from .data import MAP
from .rhythm_features import features_for_indices

def extended_features(root,data,indices,gap):
 root=Path(root);indices=np.asarray(indices);basic=features_for_indices(root,data,indices);wave=np.empty((len(indices),80));difference=np.empty_like(wave);context=np.empty((len(indices),16));locations=np.rint(np.linspace(60,300,80)).astype(int)
 for rec in sorted(set(data['record'][indices])):
  loc=np.flatnonzero(data['record'][indices]==rec);ix=indices[loc];h=wfdb.rdheader(str(root/'data'/rec));lead=h.sig_name.index('MLII') if 'MLII' in h.sig_name else 0;signal=wfdb.rdrecord(str(root/'data'/rec),channels=[lead]).p_signal[:,0];ann=wfdb.rdann(str(root/'data'/rec),'atr');peaks=np.asarray([p for p,s in zip(ann.sample,ann.symbol) if s in MAP]);rr=np.r_[288,np.diff(peaks)].astype(float);history=[]
  for position,i in zip(loc,ix):
   p=int(data['peak'][i]);j=int(np.searchsorted(peaks,p));assert peaks[j]==p;prior=rr[max(0,j-32):j];prior=prior if len(prior) else np.array([288.]);current=rr[j];quantiles=np.quantile(prior,[.1,.25,.5,.75,.9,1.]);last=np.r_[np.full(8,288.),rr[:j+1]][-8:];context[position]=np.r_[current/np.maximum(quantiles,1),current/max(prior.mean(),1),np.std(prior)/max(prior.mean(),1),last/max(prior.max(),1)]
   x=signal[p-140:p+220].astype(float);x-=np.median(np.r_[x[:60],x[-60:]]);center=x[120:161];pk=float(center[np.argmax(np.abs(center))]);norm=max(abs(pk),1e-6);v=np.clip(x[locations]/norm*np.sign(pk),-5,5);template=np.median(np.stack(history[-16:]),axis=0) if history else v;wave[position]=v;difference[position]=v-template;history.append(v)
 X=np.c_[gap,basic,context,wave,difference]
 if not np.isfinite(X).all():raise ValueError('Invalid extended features')
 return X
