from pathlib import Path
import argparse,json,csv,hashlib
import numpy as np
from qcare.frontend import cwt_maps,quantize_map
from qcare.program import read_hex,export_vectors
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--export',default='xilinx/rtl/qnn');p.add_argument('--out',default='experiments/afe_adc_checked');a=p.parse_args()
source=root/'data/2.csv';raw=np.loadtxt(source,delimiter=',',skiprows=1);valid=np.isfinite(raw).all(1);raw=raw[valid]
t,inv=np.unique(raw[:,0],return_inverse=True);v=np.bincount(inv,weights=raw[:,1])/np.bincount(inv)
# Canonical corrected ADC clock: do not append an off-grid final source sample.
ta=t[0]+np.arange(int(np.floor((t[-1]-t[0])*500+1e-9))+1)/500
ideal=np.interp(ta,t,v);code=np.clip(np.floor((ideal+1)/2*4095+.5),0,4095).astype(np.int64);reconstructed=-1+code*2/4095
model_t=ta[0]+np.arange(int(np.floor((ta[-1]-ta[0])*360+1e-9))+1)/360
model_v=np.interp(model_t,ta,reconstructed);candidates=np.arange(140,len(model_v)-219);peak=int(candidates[np.abs(model_v[candidates]-np.median(model_v)).argmax()]);beat=model_v[peak-140:peak+220]
M=cwt_maps(beat);maps=quantize_map(M)
export=root/a.export;ops=read_hex(export/'gen/qnn_prog.mem',16).reshape(-1,18);w=read_hex(export/'gen/qnn_w.mem',16);b=read_hex(export/'gen/qnn_b.mem',16)
out=root/a.out;out.mkdir(parents=True,exist_ok=True)
export_vectors(out,maps,ops,w,b,[{'source':'data/2.csv','heuristic_peak_sample_360':peak,'heuristic_peak_time_s':float(model_t[peak]),'label':None,'not_MIT_BIH_annotation':True}])
with open(out/'adc_12bit_500sps.csv','w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['sample_index','time_s','ideal_differential_V','adc_code_decimal','reconstructed_V']);writer.writerows(zip(np.arange(len(ta)),ta,ideal,code,reconstructed))
np.savetxt(out/'segment_360.csv',np.c_[model_t[peak-140:peak+220],beat],delimiter=',',header='time_s,adc_reconstructed_V',comments='')
np.savetxt(out/'cwt_map.csv',M[0,0],delimiter=',')
# Reproduce the original MATLAB path separately, including its off-grid rule
# and legacy 127/round-away quantizer, to diagnose the existing frozen vector.
legacy_t=ta.copy()
if legacy_t[-1]<t[-1]-1e-10:legacy_t=np.r_[legacy_t,t[-1]]
lv=np.interp(legacy_t,t,v);lc=np.clip(np.floor((lv+1)/2*4095+.5),0,4095);lr=-1+lc*2/4095
lt=legacy_t[0]+np.arange(int(np.floor((legacy_t[-1]-legacy_t[0])*360+1e-9))+1)/360;lm=np.interp(lt,legacy_t,lr);cand=np.arange(140,len(lm)-219);lp=int(cand[np.abs(lm[cand]-np.median(lm)).argmax()]);lmap=cwt_maps(lm[lp-140:lp+220])
old_quant=np.clip(np.sign(lmap)*np.floor(np.abs(lmap)*127+.5),-128,127).astype(np.int8)
frozen=read_hex(root/'xilinx/rtl/qnn/vec/afe_qnn_vec_in.mem',8).reshape(1,1,60,60)
difference=old_quant.astype(int)-frozen.astype(int)
provided=np.genfromtxt(root/'data/adc_test_vector_2_12bit_500sps.csv',delimiter=',',skip_header=1,usecols=(1,2,3))
# Evaluate codes at the supplied timestamps as well as the canonical ADC clock.
provided_v=np.interp(provided[:,0],t,v);provided_code=np.clip(np.floor((provided_v+1)/2*4095+.5),0,4095)
report={'status':'PASS','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_rows':len(raw),'duplicate_timestamps_averaged':len(raw)-len(t),'duration_s':float(t[-1]-t[0]),'adc':{'bits':12,'encoding':'offset-binary endpoint-inclusive behavioral model','range_V':[-1,1],'sample_rate_hz':500,'quantization':'clip(floor((v+1)/2*4095+0.5),0,4095)','LSB_V':2/4095,'codes_minmax':[int(code.min()),int(code.max())],'rail_codes':int(np.sum((code==0)|(code==4095))),'provisional_not_physical_ADC':True},'model_fs_hz':360,'resampling':'linear interpolation, same legacy method; no additional antialias claim','segment':{'pre':140,'post':219,'peak_index_0based':peak,'peak_time_s':float(model_t[peak]),'peak_method':'largest absolute deviation from whole-segment median within valid candidates, noncausal heuristic without annotations'},'memory_order':'scale row-major, signed bytes, 3600 tokens','canonical_quantizer':'floor(128*M+0.5), saturation [-128,127]','legacy_vector_replay':{'status':'PASS' if np.array_equal(old_quant,frozen) else 'FAIL','different_tokens':int(np.count_nonzero(difference)),'maximum_code_difference':int(np.max(np.abs(difference))),'baseline_vector_preserved':True},'supplied_ADC_CSV_replay':{'status':'PASS' if np.array_equal(provided_code,provided[:,2]) else 'FAIL','different_codes':int(np.count_nonzero(provided_code!=provided[:,2])),'samples':len(provided)},'classification_ground_truth':'unavailable; this is co-simulation, not DS2 accuracy','model_export':str(export.relative_to(root)),'model_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [export/'gen/qnn_prog.mem',export/'gen/qnn_w.mem',export/'gen/qnn_b.mem']}}
report['status']='PASS' if report['legacy_vector_replay']['status']=='PASS' and report['supplied_ADC_CSV_replay']['status']=='PASS' else 'FAIL'
(out/'adapter_report.json').write_text(json.dumps(report,indent=2));(root/'reports/cadence_adapter.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if report['status']!='PASS':raise SystemExit('Cadence replay differs; inspect adapter_report.json')
