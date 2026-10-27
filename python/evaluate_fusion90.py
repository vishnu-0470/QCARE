"""One DS2 follow-up evaluation of the DS1-selected, frozen fusion model."""
from pathlib import Path
import json,csv,time
import numpy as np,torch
from train_current import load_checkpoint
from qcare.model import CurrentIRN
from qcare.metrics import score,cluster_bootstrap
from qcare.provenance import verify_prepared,digest,verify_checkpoint_preparation
from qcare.rhythm_features import features_for_indices
from qcare.fusion_deploy import export_fusion,head_forward,auxiliary_codes
from qcare.program import run_program,write_hex
r=Path(__file__).resolve().parents[1];run=r/'experiments/accuracy90_fusion_mlp';out=run/'DS2_followup';out.mkdir(parents=True,exist_ok=True)
if (out/'EVALUATED.json').exists():raise RuntimeError('Follow-up DS2 already evaluated; preserve results')
f=json.loads((run/'FROZEN.json').read_text());assert digest(run/f['checkpoint'])==f['sha256'];ck=load_checkpoint(run/f['checkpoint'],weights_only=True,map_location='cpu');prep=verify_prepared(r);verify_checkpoint_preparation(ck,prep)
# Choose among neural candidates on DS1 evidence only, before indexing DS2.
selected=ck['head'];assert selected['metrics']['accuracy']>.90 and selected['metrics']['macro_f1']>=.30
(run/'FINAL_SELECTION.json').write_text(json.dumps({'status':'PASS','selected_family':'nonlinear quantized fusion','reason':'DS1-only best eligible macro-F1 across declared head trials','DS1_validation_accuracy':selected['metrics']['accuracy'],'DS1_validation_macro_f1':selected['metrics']['macro_f1'],'previous_DS2_results_known':True,'frozen_checkpoint_sha256':f['sha256']},indent=2))
ops,w,b,params,contract=export_fusion(r,ck,run/'export')
d=dict(np.load(r/'experiments/cache/beats.npz'));maps=np.load(r/'experiments/cache/maps.npy',mmap_mode='r');test=np.flatnonzero(np.isin(d['record'],prep['DS2']));assert len(test)==49688 and len(set(d['record'][test]))==22
begin=time.time();raw=features_for_indices(r,d,test);aux=auxiliary_codes(raw,ck['config'])
baseck=load_checkpoint(r/'experiments/ds1_current/best_qat.pt',weights_only=True,map_location='cpu');model=CurrentIRN();model.load_state_dict(baseck['state_dict']);model.configure_qat(baseck['E'],baseck['WE']);model.double().eval();torch.set_num_threads(1);gap=[]
with torch.no_grad():
 for j in range(0,len(test),200):
  _,trace=model(torch.tensor(maps[test[j:j+200]].astype(np.float64))/128,True);gap.append(np.rint(trace['gap'].numpy()/2.**contract['input_exponent']).astype(np.int64))
  if j%10000==0:print('DS2_INFERENCE',j,len(test),round(time.time()-begin,1),flush=True)
gap=np.concatenate(gap);logits,hidden=head_forward(gap,aux,params,contract['hidden_requant_shift']);pred=logits.argmax(1);labels=d['label'][test];m=score(labels,pred)
# Verify ten labeled vectors through the independent full microcode program.
selected_rows=[]
for c in range(5):selected_rows.extend(np.flatnonzero(labels==c)[:2].tolist())
vin=[];expected=[];identities=[]
for j in selected_rows:
 oracle,tr=run_program(maps[test[j]],ops,w,b,True,initial_memory={32064:aux[j]});np.testing.assert_array_equal(oracle,logits[j]);np.testing.assert_array_equal(tr[-2].reshape(-1),hidden[j]);np.testing.assert_array_equal(tr[-3].reshape(-1),gap[j])
 vin.extend(np.r_[maps[test[j]].ravel(),aux[j]].tolist());expected.append(' '.join(map(str,[int(pred[j]),*logits[j].tolist()])))
 identities.append({'record':str(d['record'][test[j]]),'R_sample_0based':int(d['peak'][test[j]]),'annotation_index':int(d['annotation_index'][test[j]]),'true_class':int(labels[j]),'predicted_class':int(pred[j]),'auxiliary_codes':aux[j].tolist()})
write_hex(run/'export/vec/qnn_vec_in.mem',vin,8);(run/'export/vec/qnn_vec_exp.txt').write_text('\n'.join(expected)+'\n');(run/'export/vec/identities.json').write_text(json.dumps(identities,indent=2))
m.update(status='PASS',target_accuracy_strictly_greater_than=.90,target_status='PASS' if m['accuracy']>.90 else 'FAIL',protocol='follow-up conventional DS1/DS2; DS2 used in earlier reported experiment, not a new pristine test set',previous_DS2_results_known=True,DS2_used_for_this_followup_fit_selection_normalization=False,records=prep['DS2'],fit_records=prep['train_records'],validation_records=prep['validation_records'],boundary_exclusions=24,uncertainty=cluster_bootstrap(labels,pred,d['record'][test]),checkpoint_sha256=f['sha256'],architecture=contract,scope='quantized fusion software evaluation; RTL replay reported separately',seconds=time.time()-begin)
np.savez_compressed(out/'predictions.npz',indices=test,record=d['record'][test],peak=d['peak'][test],labels=labels,predictions=pred,integer_logits=logits,auxiliary_codes=aux,gap_codes=gap)
with open(out/'predictions.csv','w',newline='') as file:
 writer=csv.writer(file);writer.writerow(['record','R_sample_0based','true_class','predicted_class',*['logit_'+c for c in ['N','S','V','F','Q']]])
 for j,i in enumerate(test):writer.writerow([d['record'][i],int(d['peak'][i]),int(labels[j]),int(pred[j]),*logits[j].tolist()])
(out/'metrics.json').write_text(json.dumps(m,indent=2));(r/'reports/accuracy90/DS2_metrics.json').write_text(json.dumps(m,indent=2));(out/'EVALUATED.json').write_text(json.dumps({'status':'PASS','checkpoint_sha256':f['sha256'],'evaluation_completed_once':True,'previous_DS2_results_known':True,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'target_status':m['target_status'],'ten_full_integer_oracle_vectors':'PASS'},indent=2));print('DS2_RESULT',json.dumps({k:m[k] for k in ['accuracy','macro_f1','target_status','per_class']}),flush=True)
