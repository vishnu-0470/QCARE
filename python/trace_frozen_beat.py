"""Trace an already-scored real beat; validate export boundaries without fitting."""
from pathlib import Path
import json,hashlib
import numpy as np,torch,wfdb
from scipy.io import loadmat,savemat
from qcare.data import sha,MAP,CLASSES
from qcare.frontend import cwt_maps,quantize_map
from qcare.program import read_hex,run_program
from qcare.model import CurrentIRN
from train_current import load_checkpoint
r=Path(__file__).resolve().parents[1];run=r/'experiments/ds1_current';ex=run/'export';out=r/'experiments/beat_trace';out.mkdir(exist_ok=True,parents=True)
identity=json.loads((ex/'vec/identities.json').read_text())[0];rec=identity['record'];ann=wfdb.rdann(str(r/'data'/rec),'atr');j=identity['annotation_index'];peak=identity['R_sample_0based']
assert int(ann.sample[j])==peak and MAP[ann.symbol[j]]==identity['label_0based']
h=wfdb.rdheader(str(r/'data'/rec));lead=h.sig_name.index('MLII') if 'MLII' in h.sig_name else 0
sig=wfdb.rdrecord(str(r/'data'/rec),channels=[lead]).p_signal[:,0];beat=sig[peak-140:peak+220];M=cwt_maps(beat);q=quantize_map(M)[0]
vin=read_hex(ex/'vec/qnn_vec_in.mem',8).reshape(-1,1,60,60);np.testing.assert_array_equal(q,vin[0])
ops=read_hex(ex/'gen/qnn_prog.mem',16).reshape(-1,18);w=read_hex(ex/'gen/qnn_w.mem',16);b=read_hex(ex/'gen/qnn_b.mem',16)
assert len(w)==9144 and len(b)==149 and w.min()>=-8 and w.max()<=7
wide=[int(v,16) for v in (ex/'gen/qnn_prog.mem.wide').read_text().split()]
for row,word in zip(ops,wide):
 assert [int(v)&65535 for v in row]==[(word>>(16*k))&65535 for k in range(18)]
logits,trace=run_program(q,ops,w,b,True);gold=np.loadtxt(ex/'vec/qnn_vec_exp.txt',dtype=np.int64);np.testing.assert_array_equal(logits,gold[0,1:]);assert logits.argmax()==gold[0,0]
# Round-trip the frozen MATLAB struct and ensure its integer weights/biases
# have exactly the same order as the hardware ROMs. No MATLAB execution claim.
mat=loadmat(run/'frozen_model.mat',struct_as_record=False,squeeze_me=False);Q=mat['Qm'][0,0];wm=[];bm=[]
for n in ['ds1','irn1','irn2','ds2','irn3','irn4','fc']:
 l=getattr(Q,n)[0,0]
 for suffix in ([''] if n in ('ds1','ds2','fc') else ['1','2','3']):
  wm.extend(getattr(l,'W'+suffix).ravel(order='C'));bm.extend(getattr(l,'b'+suffix).ravel(order='C'))
np.testing.assert_array_equal(w,wm);np.testing.assert_array_equal(b,bm)
np.savez_compressed(out/'integer_operation_trace.npz',beat=beat,cwt_map=M,quantized_map=q,**{'operation_'+str(i):v for i,v in enumerate(trace)})
np.savetxt(out/'annotated_beat.csv',np.c_[np.arange(peak-140,peak+220),np.arange(-140,220)/360,beat],delimiter=',',header='record_sample_0based,time_relative_R_s,ECG_mV',comments='')
np.savetxt(out/'cwt_map.csv',M[0,0],delimiter=',')
savemat(ex/'vec/python_golden.mat',{'qmaps':vin,'expected':gold},do_compression=True)
report={'status':'PASS','identity':identity,'annotation_symbol':ann.symbol[j],'lead':h.sig_name[lead],'sample_rate_hz':360,'window_sample_inclusive':[peak-140,peak+219],'R_index_in_window_0based':140,'CWT_recomputed_from_record_matches_memory':True,'MAT_struct_roundtrip_ROM_order_matches':True,'wide_program_matches_original_18_word_fields':True,'model_checkpoint_sha256':sha(run/'best_qat.pt'),'files_sha256':{str(p.relative_to(r)):sha(p) for p in [r/'data'/(rec+'.atr'),r/'data'/(rec+'.dat'),ex/'gen/qnn_w.mem',ex/'gen/qnn_b.mem',ex/'vec/qnn_vec_in.mem']},'integer_logits':logits.tolist(),'predicted_class':CLASSES[int(logits.argmax())],'annotated_class':identity['class'],'trace_operation_shapes':[list(v.shape) for v in trace],'memory_contract':json.loads((ex/'model_contract.json').read_text()),'MATLAB_runtime':'BLOCKED: not installed; Python validates saved .mat contents only'}
(out/'trace.json').write_text(json.dumps(report,indent=2));(r/'reports/beat_trace.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
