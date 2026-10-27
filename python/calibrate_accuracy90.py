"""DS1-only prior calibration; preserves the quantized architecture and weights."""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np,torch
from train_current import load_checkpoint
from qcare.model import CurrentIRN
from qcare.metrics import score
from qcare.program import export_model,export_vectors,read_hex
from qcare.provenance import verify_prepared,digest
r=Path(__file__).resolve().parents[1];out=r/'experiments/accuracy90';out.mkdir(parents=True,exist_ok=True)
if (out/'FROZEN.json').exists():raise RuntimeError('Follow-up is frozen; preserve it')
prep=verify_prepared(r);d=dict(np.load(r/'experiments/cache/beats.npz'));maps=np.load(r/'experiments/cache/maps.npy',mmap_mode='r')
fit=np.flatnonzero(np.isin(d['record'],prep['train_records']));val=np.flatnonzero(np.isin(d['record'],prep['validation_records']))
assert not set(d['record'][fit])&set(prep['DS2']) and not set(d['record'][val])&set(prep['DS2'])
counts=np.bincount(d['label'][fit],minlength=5);sampler=np.array([min(5000,counts[0]),2000,min(5000,counts[2]),2000,2000]);natural=(counts+1)/(counts.sum()+5);sampled=sampler/sampler.sum();ratio=np.log(natural/sampled)
config={'status':'PASS','experiment':'accuracy90_followup','baseline_preserved':True,'method':'DS1 prior-shift calibration of frozen FC bias only','formula':'delta_B16 = floor(alpha * log(p_natural_DS1 / p_balanced_epoch) / output_logit_scale + 0.5)','natural_prior_additive_smoothing':1,'alpha_grid':np.arange(0,2.501,.125).tolist(),'selection':'maximum DS1 record-held-out accuracy, then macro-F1; require macro-F1 >= unadjusted validation macro-F1, V sensitivity>=0.50 and pooled non-N sensitivity>=0.20','class_order':['N','S','V','F','Q'],'fit_records':prep['train_records'],'validation_records':prep['validation_records'],'DS2_used_for_selection':False,'DS2_already_reported_in_previous_experiment':True,'architecture':'unchanged current 14conv/fourIRN, 4W/8A/B16','training_data':'real MIT-BIH only; no new ECG training examples','fit_counts':counts.tolist(),'balanced_epoch_counts':sampler.tolist(),'natural_priors':natural.tolist(),'balanced_priors':sampled.tolist(),'log_prior_ratio':ratio.tolist(),'dataset_manifest_sha256':prep['manifest_sha256'],'baseline_checkpoint_sha256':digest(r/'experiments/ds1_current/best_qat.pt'),'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__}
(out/'selection_protocol.json').write_text(json.dumps(config,indent=2))
ck=load_checkpoint(r/'experiments/ds1_current/best_qat.pt',weights_only=True,map_location='cpu');m=CurrentIRN(26,.5);m.load_state_dict(ck['state_dict']);m.configure_qat(ck['E'],ck['WE']);m.double().eval();torch.set_num_threads(2)
s=2.**(m.E['blk4']+m.WE['fc']);logits=[];begin=time.time()
with torch.no_grad():
 for j in range(0,len(val),200):logits.append(m(torch.tensor(maps[val[j:j+200]].astype(np.float64))/128).numpy())
raw=np.rint(np.concatenate(logits)/s).astype(np.int64);labels=d['label'][val];base=score(labels,raw.argmax(1));results=[]
for alpha in config['alpha_grid']:
 delta=np.floor(alpha*ratio/s+.5).astype(np.int64);z=score(labels,(raw+delta).argmax(1));cm=np.array(z['confusion_matrix']);nonN=float(np.trace(cm[1:,1:])/(cm[1:,:].sum()))
 eligible=z['macro_f1']>=base['macro_f1'] and z['per_class']['V']['sensitivity']>=.50 and nonN>=.20
 result={'alpha':alpha,'FC_bias_delta_B16':delta.tolist(),'validation_accuracy':z['accuracy'],'validation_macro_f1':z['macro_f1'],'non_N_sensitivity':nonN,'eligible':eligible,'metrics':z};results.append(result)
 print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
valid=[x for x in results if x['eligible']]
if not valid:raise RuntimeError('No calibration satisfies DS1 validation guardrails; no DS2 evaluation authorized by this protocol')
best=max(valid,key=lambda z:(z['validation_accuracy'],z['validation_macro_f1']));delta=np.array(best['FC_bias_delta_B16']);oldbias=np.floor(m.fc.b.detach().numpy()/s+.5).astype(np.int64);newbias=oldbias+delta
if np.any(newbias<-32768) or np.any(newbias>32767):raise OverflowError('FC B16 range')
with torch.no_grad():m.fc.b.copy_(torch.from_numpy(newbias.astype(np.float64)*s))
ops,w,b=export_model(m,out/'export');basew=read_hex(r/'experiments/ds1_current/export/gen/qnn_w.mem',16);baseb=read_hex(r/'experiments/ds1_current/export/gen/qnn_b.mem',16)
np.testing.assert_array_equal(w,basew);np.testing.assert_array_equal(b[:-5],baseb[:-5]);np.testing.assert_array_equal(b[-5:],newbias)
state={'state_dict':m.state_dict(),'qat':True,'E':m.E,'WE':m.WE,'config':config,'preparation':prep,'calibration':best};torch.save(state,out/'calibrated_qat.pt')
report={'status':'PASS','baseline_validation':base,'selected':best,'all_candidates':results,'seconds':time.time()-begin,'weight_ROM_unchanged':True,'only_final_five_biases_changed':True,'FC_bias_before':oldbias.tolist(),'FC_bias_after':newbias.tolist(),'source_checkpoint_sha256':config['baseline_checkpoint_sha256']}
(out/'calibration_results.json').write_text(json.dumps(report,indent=2));(r/'reports/accuracy90/calibration_results.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'DS1_validation_logits.npz',indices=val,labels=labels,integer_logits=raw)
(out/'FROZEN.json').write_text(json.dumps({'status':'PASS','checkpoint':'calibrated_qat.pt','sha256':digest(out/'calibrated_qat.pt'),'selected_alpha':best['alpha'],'DS1_only_selection':True,'DS2_not_evaluated_in_this_followup':True,'previous_DS2_results_known':True,'architecture_and_weights_preserved':True},indent=2))
print('FROZEN',json.dumps({k:v for k,v in best.items() if k!='metrics'}),flush=True)
