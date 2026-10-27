"""Train a 4W/8A fusion head on real DS1; leave the CWT/IRN backbone frozen."""
from pathlib import Path
import json,copy,time
import numpy as np,torch
from torch import nn
from train_current import load_checkpoint
from qcare.model import CurrentIRN,Layer,quant,weight_exp
from qcare.metrics import score
from qcare.program import export_model,read_hex,write_hex
from qcare.provenance import verify_prepared,digest
from qcare.rhythm_features import NAMES,features_for_indices
r=Path(__file__).resolve().parents[1];out=r/'experiments/accuracy90_fusion';out.mkdir(parents=True,exist_ok=True)
if (out/'FROZEN.json').exists():raise RuntimeError('Fusion already frozen')
prep=verify_prepared(r);d=dict(np.load(r/'experiments/cache/beats.npz'));train=np.flatnonzero(np.isin(d['record'],prep['train_records']));val=np.flatnonzero(np.isin(d['record'],prep['validation_records']));ids=np.r_[train,val]
g=np.load(r/'experiments/accuracy90_head/DS1_frozen_gap_features.npz');np.testing.assert_array_equal(g['indices'],ids);gap=g['features'];ck=load_checkpoint(r/'experiments/ds1_current/best_qat.pt',weights_only=True,map_location='cpu');m=CurrentIRN();m.load_state_dict(ck['state_dict']);m.configure_qat(ck['E'],ck['WE']);m.double().eval();torch.set_num_threads(1);ex=m.E['blk4']
f=out/'DS1_aux_features.npz'
if f.exists():z=np.load(f);np.testing.assert_array_equal(z['indices'],ids);raw=z['features']
else:raw=features_for_indices(r,d,ids);np.savez_compressed(f,indices=ids,features=raw)
mean=raw[:len(train)].mean(0);std=raw[:len(train)].std(0);std=np.maximum(std,1e-8);aux=np.clip(np.floor(((raw-mean)/std*2)/2.**ex+.5),-128,127)*2.**ex;features=np.c_[gap,aux]
X=torch.tensor(features[:len(train)],dtype=torch.float64);XV=torch.tensor(features[len(train):],dtype=torch.float64);y=torch.tensor(d['label'][train],dtype=torch.int64);yv=d['label'][val];counts=np.bincount(d['label'][train],minlength=5)
config={'status':'PASS','method':'frozen original quantized CWT/IRN GAP16 + sixteen patient-relative rhythm/morphology proxies -> FC32x5','architecture_change':'only FC input expands16to32; all14backbone convolutions unchanged','feature_names':NAMES,'auxiliary_normalization':'DS1-fit mean/std, gain2, signed8bit with same power-of-two exponent as GAP','aux_mean':mean.tolist(),'aux_std':std.tolist(),'aux_gain':2.,'aux_exponent':ex,'auxiliary_RAM_base':32064,'causal_context':'past16 unlabeled complete beats within each record; state reset each record; no future RR or annotation labels','PQRST_caveat':'width/polarity/post-R/template measures are heuristic proxies, not clinically validated delineation','fit_records':prep['train_records'],'validation_records':prep['validation_records'],'DS2_used_for_selection_or_global_normalization':False,'previous_DS2_results_known':True,'seed':226,'class_weight_exponents':[0.,.25,.5],'epochs':260,'float_epochs':60,'QAT_epochs':200,'batch_size':1024,'optimizer':'AdamW','LR_float':.02,'LR_QAT':.004,'weight_decay':.01,'label_smoothing':.02,'selection':'among validation accuracy>90% and macroF1>=.30 maximize macroF1 then accuracy; otherwise maximum accuracy','baseline_checkpoint_sha256':digest(r/'experiments/ds1_current/best_qat.pt'),'dataset_manifest_sha256':prep['manifest_sha256'],'real_ECG_only':True,'RTL_feature_boundary':'CWT and auxiliary statistics computed on host; integer engine receives both; no claim of on-FPGA feature extraction'}
(out/'config.json').write_text(json.dumps(config,indent=2));candidates=[];summaries=[];begin=time.time()
for exponent in config['class_weight_exponents']:
 torch.manual_seed(226);rng=np.random.default_rng(226);layer=nn.Linear(32,5).double()
 with torch.no_grad():layer.weight.mul_(.1);layer.bias.copy_(torch.tensor(np.log((counts+1)/(counts.sum()+5))))
 optimizer=torch.optim.AdamW(layer.parameters(),lr=.02,weight_decay=.01);cw=torch.tensor((counts.sum()/counts)**exponent,dtype=torch.float64);ew=None;hist=[];best=None
 for ep in range(260):
  if ep==60:ew=weight_exp(layer.weight);optimizer=torch.optim.AdamW(layer.parameters(),lr=.004,weight_decay=.01)
  lr=(.02 if ep<60 else .004)*(.15+.85*.5*(1+np.cos(np.pi*(ep if ep<60 else ep-60)/(59 if ep<60 else 199))))
  for group in optimizer.param_groups:group['lr']=float(lr)
  order=rng.permutation(len(train));total=0.
  for j in range(0,len(order),1024):
   ix=order[j:j+1024];w=layer.weight if ew is None else quant(layer.weight,ew,4);b=layer.bias if ew is None else quant(layer.bias,ex+ew,16);lg=torch.nn.functional.linear(X[ix],w,b);loss=torch.nn.functional.cross_entropy(lg,y[ix],weight=cw,label_smoothing=.02)
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(layer.parameters(),5.);optimizer.step()
   if ew is not None:
    with torch.no_grad():layer.weight.clamp_(-8*2.**ew,7*2.**ew)
   total+=float(loss.detach())*len(ix)
  with torch.no_grad():
   w=layer.weight if ew is None else quant(layer.weight,ew,4);b=layer.bias if ew is None else quant(layer.bias,ex+ew,16);pred=torch.nn.functional.linear(XV,w,b).argmax(1).numpy()
  metrics=score(yv,pred);entry={'class_weight_exponent':exponent,'epoch':ep,'phase':'float_head' if ew is None else 'QAT_head','loss':total/len(train),'validation_accuracy':metrics['accuracy'],'validation_macro_f1':metrics['macro_f1'],'weight_exp':ew};hist.append(entry)
  if ep%25==0 or ep==259:print(json.dumps(entry),flush=True)
  if ew is not None:
   eligible=metrics['accuracy']>.90 and metrics['macro_f1']>=.30;key=(int(eligible),metrics['macro_f1'] if eligible else metrics['accuracy'],metrics['accuracy'] if eligible else metrics['macro_f1'])
   if best is None or key>best['selection_key']:best={'selection_key':key,'state':copy.deepcopy(layer.state_dict()),'weight_exponent':ew,'metrics':metrics,'epoch':ep,'class_weight_exponent':exponent}
 (out/f'history_weight_{exponent}.json').write_text(json.dumps(hist,indent=2));summaries.append({k:v for k,v in best.items() if k!='state'});candidates.append(best)
best=max(candidates,key=lambda v:v['selection_key']);m.fc=Layer((5,32),32).double()
with torch.no_grad():m.fc.W.copy_(best['state']['weight']);m.fc.b.copy_(best['state']['bias'])
m.WE['fc']=best['weight_exponent'];ops,w,b=export_model(m,out/'export');ops[-2,4]=32;write_hex(out/'export/gen/qnn_prog.mem',ops);(out/'export/gen/qnn_prog.mem.wide').write_text(''.join(''.join(f'{int(v)&65535:04x}' for v in row[::-1])+'\n' for row in ops))
contract=json.loads((out/'export/model_contract.json').read_text());contract.update(fc_input_channels=32,auxiliary_features=16,auxiliary_RAM_base=32064,activation_ram_used=32080,host_side_auxiliary_preprocessing=True);(out/'export/model_contract.json').write_text(json.dumps(contract,indent=2))
np.testing.assert_array_equal(w[:-160],read_hex(r/'experiments/ds1_current/export/gen/qnn_w.mem',16)[:-80]);np.testing.assert_array_equal(b[:-5],read_hex(r/'experiments/ds1_current/export/gen/qnn_b.mem',16)[:-5])
xq=np.rint(features[len(train):]/2.**ex).astype(np.int64);oracle=xq@w[-160:].reshape(5,32).T+b[-5:]
with torch.no_grad():software=torch.nn.functional.linear(XV,quant(m.fc.W,m.WE['fc'],4),quant(m.fc.b,ex+m.WE['fc'],16)).numpy()
np.testing.assert_array_equal(np.rint(software/2.**(ex+m.WE['fc'])).astype(np.int64),oracle)
torch.save({'state_dict':m.state_dict(),'E':m.E,'WE':m.WE,'config':config,'preparation':prep},out/'fusion_qat.pt');report={'status':'PASS','selected':{k:v for k,v in best.items() if k!='state'},'candidates':summaries,'seconds':time.time()-begin,'all_DS1_validation_head_integer_logits_exact':True,'backbone_ROM_unchanged':True};(out/'selection_results.json').write_text(json.dumps(report,indent=2));(out/'FROZEN.json').write_text(json.dumps({'status':'PASS','checkpoint':'fusion_qat.pt','sha256':digest(out/'fusion_qat.pt'),'DS1_only_selection':True,'previous_DS2_results_known':True,'DS2_not_evaluated_in_followup':True},indent=2));print('FROZEN_FUSION',json.dumps(report['selected']),flush=True)
