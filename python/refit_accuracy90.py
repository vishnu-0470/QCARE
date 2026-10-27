"""Refit the existing 4-bit FC head, using frozen quantized real-DS1 features."""
from pathlib import Path
import json,time,copy
import numpy as np,torch
from torch import nn
from train_current import load_checkpoint
from qcare.model import CurrentIRN,quant,weight_exp
from qcare.metrics import score
from qcare.program import export_model,read_hex
from qcare.provenance import verify_prepared,digest
r=Path(__file__).resolve().parents[1];out=r/'experiments/accuracy90_head';out.mkdir(parents=True,exist_ok=True);cache=r/'experiments/cache'
if (out/'FROZEN.json').exists():raise RuntimeError('Head follow-up already frozen')
prep=verify_prepared(r);d=dict(np.load(cache/'beats.npz'));maps=np.load(cache/'maps.npy',mmap_mode='r');train=np.flatnonzero(np.isin(d['record'],prep['train_records']));val=np.flatnonzero(np.isin(d['record'],prep['validation_records']));ids=np.r_[train,val]
ck=load_checkpoint(r/'experiments/ds1_current/best_qat.pt',weights_only=True,map_location='cpu');base=CurrentIRN();base.load_state_dict(ck['state_dict']);base.configure_qat(ck['E'],ck['WE']);base.double().eval();torch.set_num_threads(1)
feature_file=out/'DS1_frozen_gap_features.npz'
if feature_file.exists():
 z=np.load(feature_file);np.testing.assert_array_equal(z['indices'],ids);features=z['features']
else:
 arrays=[];begin=time.time()
 with torch.no_grad():
  for j in range(0,len(ids),200):
   _,trace=base(torch.tensor(maps[ids[j:j+200]].astype(np.float64))/128,True);arrays.append(trace['gap'].numpy())
   if j%10000==0:print('DS1_FEATURES',j,len(ids),round(time.time()-begin,1),flush=True)
 features=np.concatenate(arrays);np.savez_compressed(feature_file,indices=ids,features=features,source_sha256=digest(r/'experiments/ds1_current/best_qat.pt'))
X=torch.tensor(features[:len(train)],dtype=torch.float64);XV=torch.tensor(features[len(train):],dtype=torch.float64);y=torch.tensor(d['label'][train],dtype=torch.int64);yv=d['label'][val];counts=np.bincount(d['label'][train],minlength=5);ex=base.E['blk4']
config={'status':'PASS','method':'refit only existing FC5x16; frozen original 4W/8A backbone and CWT','fit_records':prep['train_records'],'validation_records':prep['validation_records'],'DS2_used_in_selection':False,'previous_DS2_results_known':True,'seed':126,'class_weight_exponent_grid':[0.,.25,.5],'epochs_per_candidate':240,'float_head_epochs':60,'QAT_head_epochs':180,'batch_size':1024,'optimizer':'AdamW','float_initial_LR':.03,'QAT_initial_LR':.005,'weight_decay':.01,'loss':'weighted cross-entropy with label_smoothing .02','selection':'prefer validation accuracy>90% and macro-F1>=0.30; among eligible maximize macro-F1 then accuracy; otherwise maximize accuracy then macro-F1','baseline_sha256':digest(r/'experiments/ds1_current/best_qat.pt'),'real_training_only':True,'architecture_unchanged':True,'input_exponent':ex,'dataset_manifest_sha256':prep['manifest_sha256']}
(out/'config.json').write_text(json.dumps(config,indent=2))
all_results=[];global_candidates=[];start=time.time()
for exponent in config['class_weight_exponent_grid']:
 torch.manual_seed(126);rng=np.random.default_rng(126);layer=nn.Linear(16,5).double()
 with torch.no_grad():layer.weight.mul_(.1);layer.bias.copy_(torch.tensor(np.log((counts+1)/(counts.sum()+5))))
 optimizer=torch.optim.AdamW(layer.parameters(),lr=.03,weight_decay=.01);weights=torch.tensor((counts.sum()/counts)**exponent,dtype=torch.float64);ew=None;hist=[];best_for_candidate=None
 for ep in range(240):
  if ep==60:
   ew=weight_exp(layer.weight);optimizer=torch.optim.AdamW(layer.parameters(),lr=.005,weight_decay=.01)
  lr=(.03 if ep<60 else .005)*(.15+.85*.5*(1+np.cos(np.pi*(ep if ep<60 else ep-60)/(59 if ep<60 else 179))))
  for group in optimizer.param_groups:group['lr']=float(lr)
  layer.train();order=rng.permutation(len(train));total=0.
  for j in range(0,len(order),1024):
   ix=order[j:j+1024];w=layer.weight if ew is None else quant(layer.weight,ew,4);b=layer.bias if ew is None else quant(layer.bias,ex+ew,16)
   logits=torch.nn.functional.linear(X[ix],w,b);loss=torch.nn.functional.cross_entropy(logits,y[ix],weight=weights,label_smoothing=.02)
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(layer.parameters(),5.);optimizer.step()
   if ew is not None:
    with torch.no_grad():layer.weight.clamp_(-8*2.**ew,7*2.**ew)
   total+=float(loss.detach())*len(ix)
  with torch.no_grad():
   w=layer.weight if ew is None else quant(layer.weight,ew,4);b=layer.bias if ew is None else quant(layer.bias,ex+ew,16);pred=torch.nn.functional.linear(XV,w,b).argmax(1).numpy()
  m=score(yv,pred);entry={'class_weight_exponent':exponent,'epoch':ep,'phase':'float_head' if ew is None else 'QAT_head','loss':total/len(train),'validation_accuracy':m['accuracy'],'validation_macro_f1':m['macro_f1'],'fc_weight_exp':ew};hist.append(entry)
  if ep%20==0 or ep==239:print(json.dumps(entry),flush=True)
  if ew is not None:
   eligible=m['accuracy']>.90 and m['macro_f1']>=.30
   key=(int(eligible),m['macro_f1'] if eligible else m['accuracy'],m['accuracy'] if eligible else m['macro_f1'])
   if best_for_candidate is None or key>best_for_candidate['selection_key']:
    best_for_candidate={'selection_key':key,'weight_exponent':ew,'state':copy.deepcopy(layer.state_dict()),'metrics':m,'epoch':ep,'class_weight_exponent':exponent}
 (out/f'history_weight_{exponent}.json').write_text(json.dumps(hist,indent=2));all_results.append({k:v for k,v in best_for_candidate.items() if k!='state'});global_candidates.append(best_for_candidate)
best=max(global_candidates,key=lambda v:v['selection_key']);m=base
with torch.no_grad():m.fc.W.copy_(best['state']['weight']);m.fc.b.copy_(best['state']['bias'])
m.WE['fc']=best['weight_exponent'];ops,w,b=export_model(m,out/'export')
np.testing.assert_array_equal(w[:-80],read_hex(r/'experiments/ds1_current/export/gen/qnn_w.mem',16)[:-80]);np.testing.assert_array_equal(b[:-5],read_hex(r/'experiments/ds1_current/export/gen/qnn_b.mem',16)[:-5])
# Verify all saved DS1 validation head outputs using independent integer MACs.
wq=w[-80:].reshape(5,16);bq=b[-5:];xq=np.rint(features[len(train):]/2.**ex).astype(np.int64);exact=xq@wq.T+bq
with torch.no_grad():software=torch.nn.functional.linear(XV,quant(m.fc.W,m.WE['fc'],4),quant(m.fc.b,ex+m.WE['fc'],16)).numpy()
np.testing.assert_array_equal(np.rint(software/2.**(ex+m.WE['fc'])).astype(np.int64),exact)
summary={'status':'PASS','selected':{k:v for k,v in best.items() if k!='state'},'all_candidates':all_results,'seconds':time.time()-start,'backbone_weight_ROM_unchanged':True,'only_FC_weights_and_biases_changed':True,'all_DS1_validation_head_logits_bit_exact':True}
(out/'selection_results.json').write_text(json.dumps(summary,indent=2));torch.save({'state_dict':m.state_dict(),'qat':True,'E':m.E,'WE':m.WE,'config':config,'preparation':prep},out/'refit_qat.pt');(out/'FROZEN.json').write_text(json.dumps({'status':'PASS','checkpoint':'refit_qat.pt','sha256':digest(out/'refit_qat.pt'),'selection':'DS1 record-held-out validation only','previous_DS2_results_known':True,'DS2_not_evaluated_in_followup':True},indent=2));print('FROZEN_SELECTED',json.dumps(summary['selected']),flush=True)
