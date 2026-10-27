"""One held-out DS2 evaluation after checkpoint selection is complete."""
from pathlib import Path
import json,csv,sys,hashlib,copy
import numpy as np,torch,joblib
from scipy.io import savemat
from train_current import load_checkpoint
from qcare.model import CurrentIRN
from qcare.program import export_model,export_vectors,run_program
from qcare.metrics import score,cluster_bootstrap
from qcare.data import sha
from qcare.provenance import verify_prepared,verify_checkpoint_preparation
root=Path(__file__).resolve().parents[1];run=root/'experiments/ds1_current';cache=root/'experiments/cache';out=root/'experiments/ds2_evaluation';out.mkdir(exist_ok=True,parents=True)
lock=run/'FROZEN.json'
if not lock.exists():raise RuntimeError('BLOCKED: complete training and freeze before DS2 evaluation')
frozen=json.loads(lock.read_text())
if sha(run/frozen['checkpoint'])!=frozen['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
if (out/'EVALUATED.json').exists():raise RuntimeError('DS2 already evaluated; preserve this report, do not tune against it')
ck=load_checkpoint(run/'best_qat.pt',weights_only=True,map_location='cpu');model=CurrentIRN(26,.5);model.load_state_dict(ck['state_dict']);model.configure_qat(ck['E'],ck['WE']);model.eval();torch.set_num_threads(2)
# Float64 evaluation emulates exact power-of-two integer arithmetic; this
# avoids observed batch-dependent float32 convolution deviations. The frozen
# parameters/scales and checkpoint selection are unchanged.
float_latent=copy.deepcopy(model);float_latent.qat=False
model=model.double()
prep=verify_prepared(root);verify_checkpoint_preparation(ck,prep);data=dict(np.load(cache/'beats.npz'));maps=np.load(cache/'maps.npy',mmap_mode='r');test=np.flatnonzero(np.isin(data['record'],prep['DS2']));labels=data['label'][test]
if set(data['record'][test])!=set(prep['DS2']):raise ValueError('Incomplete DS2')
# Export only the pre-frozen checkpoint, leaving original xilinx/rtl/qnn/gen untouched.
export=run/'export';ops,w,b=export_model(model,export)
selected=[]
for c in range(5):selected.extend(test[labels==c][:2].tolist())
identities=[{'record':str(data['record'][i]),'annotation_index':int(data['annotation_index'][i]),'R_sample_0based':int(data['peak'][i]),'label_0based':int(data['label'][i]),'class':['N','S','V','F','Q'][data['label'][i]]} for i in selected]
export_vectors(export,maps[np.array(selected)],ops,w,b,identities)
qat_logits=[];float_logits=[];warm_logits=[]
warm=load_checkpoint(run/'best_float.pt',weights_only=True,map_location='cpu');baseline=CurrentIRN(26,.5);baseline.load_state_dict(warm['state_dict']);baseline.eval()
with torch.no_grad():
 for start in range(0,len(test),200):
  ix=test[start:start+200];x=torch.tensor(maps[ix].astype(np.float32))/128
  qat_logits.append(model(x.double()).numpy());float_logits.append(float_latent(x).numpy());warm_logits.append(baseline(x).numpy())
model.qat=True;qat_logits=np.concatenate(qat_logits);float_logits=np.concatenate(float_logits);warm_logits=np.concatenate(warm_logits)
# The independent oracle checks every intermediate tensor on the selected real
# beats. Its final integer logits are also compared to batched QAT inference.
lookup={int(v):i for i,v in enumerate(test)}
tensor_names=['ds1','irn1_A','irn1_BM','irn1_B1P','irn1','irn2_A','irn2_BM','irn2_B1P','irn2','pool','ds2','irn3_A','irn3_BM','irn3_B1P','irn3','irn4_A','irn4_BM','irn4_B1P','irn4','gap','logits']
for i in selected:
 expected,itrace=run_program(maps[i],ops,w,b,True)
 with torch.no_grad(): _,ttrace=model(torch.tensor(maps[i].astype(np.float64))[None]/128,True)
 for name,value in zip(tensor_names,itrace):
  exp=model.E['ds1'] if name=='ds1' else model.E['ds2'] if name=='ds2' else model.E['blk2'] if name=='pool' else model.E['blk4'] if name=='gap' else model.E['blk4']+model.WE['fc'] if name=='logits' else model.E['blk'+name[3]]
  np.testing.assert_array_equal(np.rint(ttrace[name][0].numpy()/2.**exp).astype(np.int64).reshape(value.shape),value)
 got=np.rint(qat_logits[lookup[i]]/2.0**(model.E['blk4']+model.WE['fc'])).astype(np.int64)
 np.testing.assert_array_equal(got,expected)
rr_model=joblib.load(root/'experiments/rr_baseline/model.joblib');rr_pred=rr_model.predict(data['rr'][test])
predictions={'quantized_current_irn':qat_logits.argmax(1),'float_same_latent_parameters':float_logits.argmax(1),'float_warm_start_checkpoint':warm_logits.argmax(1),'RR_logistic_baseline':rr_pred}
reports={}
for name,pred in predictions.items():
 m=score(labels,pred);m.update(quantized_evaluation_arithmetic='float64 power-of-two arithmetic checked against independent integer oracle' if name=='quantized_current_irn' else 'float64 sklearn StandardScaler/logistic regression' if name=='RR_logistic_baseline' else 'float32 software',status='PASS',meaning='executed evaluation, not an accuracy-target pass',records=prep['DS2'],included_records=len(prep['DS2']),included_beats=len(test),boundary_exclusions=sum(prep['edge_exclusions_per_record'][r] for r in prep['DS2']),protocol='conventional DS1 fit subset / DS1 record-held-out validation / full DS2 test',same_subject_caveat='201 DS1 / 202 DS2 are one person; do not describe as strict subject-disjoint',uncertainty=cluster_bootstrap(labels,pred,data['record'][test]))
 reports[name]=m
(out/'metrics.json').write_text(json.dumps(reports,indent=2))
np.savez_compressed(out/'predictions.npz',indices=test,record=data['record'][test],peak=data['peak'][test],labels=labels,qat_logits=qat_logits,float_logits=float_logits,warm_logits=warm_logits,**predictions)
with open(out/'predictions.csv','w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['record','R_sample','true_class',*predictions])
 for j,i in enumerate(test):writer.writerow([data['record'][i],int(data['peak'][i]),int(labels[j]),*[int(v[j]) for v in predictions.values()]])
# MATLAB-compatible frozen state for the existing forward/export functions.
P={}
for name in ['ds1','irn1','irn2','ds2','irn3','irn4','fc']:
 layer=getattr(model,name);P[name]={k:np.asarray(v.detach().numpy(),dtype=np.float64) for k,v in layer.named_parameters(recurse=False)}
 for key in P[name]:
  if key.startswith('b'):P[name][key]=P[name][key].reshape(-1,1)
Qm={'e_in':-7,'E':model.E}
for name in ['ds1','ds2','fc']:
 ew=model.WE[name];ex=-7 if name=='ds1' else model.E['blk2'] if name=='ds2' else model.E['blk4'];ea=ex+ew
 Qm[name]={'W':np.clip(np.floor(P[name]['W']/2.**ew+.5),-8,7),'eW':ew,'b':np.floor(P[name]['b']/2.**ea+.5),'eA':ea}
for n in range(1,5):
 name='irn'+str(n);ex=model.E['ds1'] if n==1 else model.E['blk1'] if n==2 else model.E['ds2'] if n==3 else model.E['blk3'];eb=model.E['blk'+str(n)];q={'ex':ex,'eb':eb}
 for k in range(1,4):
  ew=model.WE[name+'_W'+str(k)];ea=(ex if k==1 else eb)+ew;q['W'+str(k)]=np.clip(np.floor(P[name]['W'+str(k)]/2.**ew+.5),-8,7);q['eW'+str(k)]=ew;q['eA'+str(k)]=ea;q['b'+str(k)]=np.floor(P[name]['b'+str(k)]/2.**ea+.5)
 Qm[name]=q
savemat(run/'frozen_model.mat',{'P':P,'E':model.E,'eWfix':model.WE,'Qm':Qm},do_compression=True)
(out/'EVALUATED.json').write_text(json.dumps({'status':'PASS','frozen_checkpoint_sha256':frozen['checkpoint_sha256'],'selection_completed_before_evaluation':True,'test_indices_sha256':hashlib.sha256(test.tobytes()).hexdigest(),'same_scored_beats_for_RR_and_QNN':True,'no_DS2_fit_or_tuning':True,'oracle_parity':'all 21 operation tensors on 10 real DS2 beats; batched final logits exact on same 10', 'checkpoint_unchanged_after_numeric_fix':True,'primary_quantized_accuracy':reports['quantized_current_irn']['accuracy'],'primary_macro_f1':reports['quantized_current_irn']['macro_f1']},indent=2))
(root/'reports/ds2_metrics.json').write_text(json.dumps(reports,indent=2))
print(json.dumps({k:{'accuracy':v['accuracy'],'macro_f1':v['macro_f1'],'beats':v['included_beats']} for k,v in reports.items()},indent=2))
