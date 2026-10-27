"""Deterministic beat/rhythm proxies, with causal unlabeled patient context."""
from pathlib import Path
import numpy as np,wfdb
from scipy.ndimage import gaussian_filter1d
NAMES=['previous_RR_s','previous_RR_over_past_median','previous_over_preceding_RR','RR_difference_over_past_median','past_RR_coefficient_of_variation','QRS_envelope_width20_samples','QRS_envelope_width50_samples','QRS_peak_polarity','QRS_peak_to_peak_mV','QRS_max_slope_over_amplitude','QRS_signed_area_over_amplitude_s','width20_over_past16_median','amplitude_over_past16_median','past16_template_cosine_distance','past16_template_mean_abs_difference','post_R_deflection_over_R_amplitude']

def record_features(signal,peaks,rr):
 history=[];widths=[];amplitudes=[];rows=[]
 for p,intervals in zip(peaks,rr):
  beat=np.asarray(signal[int(p)-140:int(p)+220],dtype=np.float64);assert len(beat)==360
  x=beat-np.median(np.r_[beat[:60],beat[-60:]]);q=x[100:181];idx=120+int(np.argmax(np.abs(x[120:161])));amp=max(float(abs(x[idx])),1e-6);env=gaussian_filter1d(np.abs(x),2.);peak_env=float(env[idx])
  def width(frac):
   lo=idx;hi=idx
   while lo>80 and env[lo-1]>=frac*peak_env:lo-=1
   while hi<200 and env[hi+1]>=frac*peak_env:hi+=1
   return hi-lo+1
  w20=width(.2);w50=width(.5);z=q/amp;template=np.mean(history[-16:],axis=0) if history else z
  norm=max(float(np.linalg.norm(z)*np.linalg.norm(template)),1e-9);distance=1-float(z@template)/norm
  width_relative=w20/(np.median(widths[-16:]) if widths else w20);amp_relative=amp/(np.median(amplitudes[-16:]) if amplitudes else amp)
  tail=x[170:261];post=float(tail[np.argmax(np.abs(tail))])/amp
  rows.append([intervals[0],intervals[2],intervals[3],intervals[4],intervals[5],w20,w50,float(np.sign(x[idx])),float(np.ptp(q)),float(np.max(np.abs(np.diff(q))))/amp,float(q.sum())/(amp*360),width_relative,amp_relative,distance,float(np.mean(np.abs(z-template))),post])
  history.append(z);widths.append(w20);amplitudes.append(amp)
 return np.asarray(rows,dtype=np.float64)

def features_for_indices(root,data,indices):
 root=Path(root);indices=np.asarray(indices);out=np.empty((len(indices),len(NAMES)),dtype=np.float64)
 for rec in sorted(set(data['record'][indices])):
  loc=np.flatnonzero(data['record'][indices]==rec);ix=indices[loc];assert np.all(np.diff(data['peak'][ix])>=0)
  h=wfdb.rdheader(str(root/'data'/rec));lead=h.sig_name.index('MLII') if 'MLII' in h.sig_name else 0;signal=wfdb.rdrecord(str(root/'data'/rec),channels=[lead]).p_signal[:,0]
  out[loc]=record_features(signal,data['peak'][ix],data['rr'][ix])
 if not np.isfinite(out).all():raise ValueError('Nonfinite rhythm/morphology feature')
 return out
