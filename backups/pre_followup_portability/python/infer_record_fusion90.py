"""Replay an annotated real MIT-BIH beat without requiring prepared caches."""
from pathlib import Path
import argparse,json
import numpy as np,torch,wfdb
from qcare.data import MAP,CLASSES
from qcare.rhythm_features import record_features
from qcare.frontend import cwt_maps,quantize_map
from qcare.fusion_deploy import auxiliary_codes,quant_head,head_forward
from qcare.program import read_hex,run_program,write_hex
from qcare.model import CurrentIRN
from qcare.provenance import digest,check_file
from train_current import load_checkpoint
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--record',default='100');p.add_argument('--beat-index',type=int,default=0);p.add_argument('--output');args=p.parse_args();run=r/'experiments/accuracy90_fusion_mlp';f=json.loads((run/'FROZEN.json').read_text());check_file(run/f['checkpoint'],f['sha256']);ck=load_checkpoint(run/f['checkpoint'],weights_only=True,map_location='cpu');cfg=ck['config'];manifest=json.loads((r/'reports/dataset_manifest.json').read_text());entry=next(x for x in manifest['records'] if x['record']==args.record)
for ext,info in entry['files'].items():check_file(r/'data'/(args.record+'.'+ext),info['sha256'])
h=wfdb.rdheader(str(r/'data'/args.record));lead=h.sig_name.index('MLII') if 'MLII' in h.sig_name else 0;signal=wfdb.rdrecord(str(r/'data'/args.record),channels=[lead]).p_signal[:,0];a=wfdb.rdann(str(r/'data'/args.record),'atr');selected=[(i,int(v),MAP[s]) for i,(v,s) in enumerate(zip(a.sample,a.symbol)) if s in MAP];peaks=[];rrs=[];labels=[];annotation_indices=[];history=[];previous=None;last=288
for annidx,v,label in selected:
 rr=v-previous if previous is not None else 288;median=float(np.median(history[-16:])) if history else 288.;std=float(np.std(history[-16:])) if len(history)>1 else 0.
 feature=[rr/360,last/360,rr/max(median,1),rr/max(last,1),(rr-last)/max(median,1),std/max(median,1)]
 if v>=140 and v+219<len(signal):peaks.append(v);rrs.append(feature);labels.append(label);annotation_indices.append(annidx)
 previous=v;last=rr;history.append(rr)
if not 0<=args.beat_index<len(peaks):raise ValueError('Beat index outside complete-window range')
# Use every preceding complete beat for exactly the declared causal state.
features=record_features(signal,peaks[:args.beat_index+1],rrs[:args.beat_index+1]);aux=auxiliary_codes(features[-1:],cfg)[0];peak=peaks[args.beat_index];beat=signal[peak-140:peak+220];M=cwt_maps(beat);q=quantize_map(M)[0];gen=run/'export/gen';ops=read_hex(gen/'qnn_prog.mem',16).reshape(-1,18);w=read_hex(gen/'qnn_w.mem',16);b=read_hex(gen/'qnn_b.mem',16);oracle,trace=run_program(q,ops,w,b,True,initial_memory={32064:aux})
baseck=load_checkpoint(r/'experiments/ds1_current/best_qat.pt',weights_only=True,map_location='cpu');model=CurrentIRN();model.load_state_dict(baseck['state_dict']);model.configure_qat(baseck['E'],baseck['WE']);model.double().eval();torch.set_num_threads(1)
with torch.no_grad():_,t=model(torch.tensor(q.astype(np.float64))[None]/128,True)
head=ck['head'];gap=np.rint(t['gap'].numpy()/2.**cfg['aux_exponent']).astype(np.int64);params=quant_head(head['state'],cfg['aux_exponent'],head['ew1'],head['eh'],head['ew2']);lg,_=head_forward(gap,aux[None],params,head['eh']-cfg['aux_exponent']-head['ew1']);np.testing.assert_array_equal(oracle,lg[0])
out=r/(args.output or f'experiments/replay/record{args.record}_beat{args.beat_index}');(out/'vec').mkdir(exist_ok=True,parents=True);write_hex(out/'vec/qnn_vec_in.mem',np.r_[q.ravel(),aux],8);(out/'vec/qnn_vec_exp.txt').write_text(' '.join(map(str,[int(oracle.argmax()),*oracle.tolist()]))+'\n');np.savetxt(out/'cwt_map.csv',M[0,0],delimiter=',');np.savetxt(out/'beat.csv',np.c_[np.arange(-140,220)/360,beat],delimiter=',',header='time_relative_R_s,ECG_mV',comments='')
result={'status':'PASS','record':args.record,'beat_index':args.beat_index,'annotation_index':annotation_indices[args.beat_index],'R_sample_0based':peak,'annotated_class':CLASSES[labels[args.beat_index]],'predicted_class':CLASSES[int(oracle.argmax())],'integer_logits':oracle.tolist(),'model_sha256':f['sha256'],'independent_integer_oracle':'PASS','output_directory':str(out.relative_to(r))};(out/'inference.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
