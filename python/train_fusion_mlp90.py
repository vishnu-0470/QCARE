"""DS1-only nonlinear fusion head: signed8 inputs -> 4W dense32 -> ReLU8 -> FC5."""
from pathlib import Path
import json,time,copy
import numpy as np,torch,joblib
from torch import nn
from qcare.model import quant,weight_exp
from qcare.metrics import score
from qcare.provenance import verify_prepared,digest
r=Path(__file__).resolve().parents[1];out=r/'experiments/accuracy90_fusion_mlp';out.mkdir(parents=True,exist_ok=True)
if (out/'FROZEN.json').exists():raise RuntimeError('Nonlinear fusion already frozen')
prep=verify_prepared(r);d=np.load(r/'experiments/cache/beats.npz');g=np.load(r/'experiments/accuracy90_head/DS1_frozen_gap_features.npz');a=np.load(r/'experiments/accuracy90_fusion/DS1_aux_features.npz');np.testing.assert_array_equal(g['indices'],a['indices']);ids=g['indices'];fit=np.isin(d['record'][ids],prep['train_records']);gap=g['features'];raw=a['features'];fcfg=json.loads((r/'experiments/accuracy90_fusion/config.json').read_text());ex=fcfg['aux_exponent'];mean=np.array(fcfg['aux_mean']);std=np.array(fcfg['aux_std']);aux=np.clip(np.floor((raw-mean)/std*2/2.**ex+.5),-128,127)*2.**ex;features=np.c_[gap,aux];labels=d['label'][ids];counts=np.bincount(labels[fit],minlength=5);torch.set_num_threads(1)
X=torch.tensor(features[fit],dtype=torch.float64);XV=torch.tensor(features[~fit],dtype=torch.float64);Y=torch.tensor(labels[fit],dtype=torch.int64);yv=labels[~fit]
teacher=joblib.load(r/'experiments/accuracy90_diagnostics/teacher.joblib');teacherprob=torch.tensor(teacher.predict_proba(np.c_[gap[fit],raw[fit]]),dtype=torch.float64)
config={**fcfg,'method':'frozen original GAP16 + signed8 auxiliary16 -> 4W pointwise32/ReLU8 -> 4W FC5','architecture_change':'14original convolutions unchanged; one32x32 pointwise layer added before FC','hidden_channels':32,'epochs':260,'float_epochs':80,'QAT_epochs':180,'candidates':[{'class_weight_exponent':v,'distillation_weight':t} for v in [0.,.25] for t in [0.,.35]],'optimizer':'AdamW','LR_float':.006,'LR_QAT':.0015,'weight_decay':.05,'dropout_train':.1,'label_smoothing':.01,'seed':426,'teacher_scope':'DS1-fit-only softwareHGBT, used only for soft targets on same real training beats; not deployed','teacher_sha256':digest(r/'experiments/accuracy90_diagnostics/teacher.joblib'),'selection':'prefer validation accuracy>90% and macroF1>=.30, maximize macroF1 then accuracy; otherwise accuracy then macroF1','DS2_used_for_selection_or_global_normalization':False}
(out/'config.json').write_text(json.dumps(config,indent=2))
class Head(nn.Module):
 def __init__(self):
  super().__init__();self.l1=nn.Linear(32,32).double();self.l2=nn.Linear(32,5).double();self.qat=False;self.ew1=self.ew2=self.eh=None
  with torch.no_grad():self.l2.bias.copy_(torch.tensor(np.log((counts+1)/(counts.sum()+5))))
 def forward(self,x):
  w1=self.l1.weight if not self.qat else quant(self.l1.weight,self.ew1,4);b1=self.l1.bias if not self.qat else quant(self.l1.bias,ex+self.ew1,16);h=torch.nn.functional.linear(x,w1,b1);h=quant(h,self.eh,8,True) if self.qat else torch.relu(h);h=torch.nn.functional.dropout(h,p=.1,training=self.training)
  if self.qat:h=quant(h,self.eh,8,True)
  w2=self.l2.weight if not self.qat else quant(self.l2.weight,self.ew2,4);b2=self.l2.bias if not self.qat else quant(self.l2.bias,self.eh+self.ew2,16)
  return torch.nn.functional.linear(h,w2,b2)
 @torch.no_grad()
 def start_qat(self):
  self.ew1=weight_exp(self.l1.weight);self.ew2=weight_exp(self.l2.weight);h=torch.relu(self.l1(X));v=float(torch.quantile(h.ravel(),.999));self.eh=int(np.ceil(np.log2(max(v,1e-7)/127)));self.qat=True;self.clip()
 @torch.no_grad()
 def clip(self):
  if self.qat:
   self.l1.weight.clamp_(-8*2.**self.ew1,7*2.**self.ew1);self.l2.weight.clamp_(-8*2.**self.ew2,7*2.**self.ew2)
begin=time.time();candidates=[];summaries=[]
for spec in config['candidates']:
 torch.manual_seed(426);rng=np.random.default_rng(426);model=Head();opt=torch.optim.AdamW(model.parameters(),lr=.006,weight_decay=.05);cw=torch.tensor((counts.sum()/counts)**spec['class_weight_exponent'],dtype=torch.float64);best=None;hist=[];floatbest=None
 for ep in range(260):
  if ep==80:
   model.load_state_dict(floatbest['state']);model.start_qat();opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.05)
  lr=(.006 if ep<80 else .0015)*(.1+.9*.5*(1+np.cos(np.pi*(ep if ep<80 else ep-80)/(79 if ep<80 else 179))))
  for group in opt.param_groups:group['lr']=float(lr)
  model.train();order=rng.permutation(len(X));total=0.
  for j in range(0,len(X),1024):
   ix=order[j:j+1024];logits=model(X[ix]);ce=torch.nn.functional.cross_entropy(logits,Y[ix],weight=cw,label_smoothing=.01);kd=-(teacherprob[ix]*torch.log_softmax(logits,1)).sum(1).mean();loss=(1-spec['distillation_weight'])*ce+spec['distillation_weight']*kd
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();model.clip();total+=float(loss.detach())*len(ix)
  model.eval()
  with torch.no_grad():pred=model(XV).argmax(1).numpy()
  metrics=score(yv,pred);entry={'candidate':spec,'epoch':ep,'phase':'QAT' if model.qat else 'float','loss':total/len(X),'validation_accuracy':metrics['accuracy'],'validation_macro_f1':metrics['macro_f1']};hist.append(entry)
  if ep%25==0 or ep==259:print(json.dumps(entry),flush=True)
  eligible=metrics['accuracy']>.90 and metrics['macro_f1']>=.30;key=(int(eligible),metrics['macro_f1'] if eligible else metrics['accuracy'],metrics['accuracy'] if eligible else metrics['macro_f1'])
  state={'selection_key':key,'state':copy.deepcopy(model.state_dict()),'metrics':metrics,'epoch':ep,'candidate':spec,'ew1':model.ew1,'ew2':model.ew2,'eh':model.eh}
  if not model.qat:
   if floatbest is None or key>floatbest['selection_key']:floatbest=state
  elif best is None or key>best['selection_key']:best=state
 tag=f'w{spec["class_weight_exponent"]}_kd{spec["distillation_weight"]}';(out/(tag+'_history.json')).write_text(json.dumps(hist,indent=2));torch.save(best,out/(tag+'_best.pt'));candidates.append(best);summaries.append({k:v for k,v in best.items() if k!='state'})
best=max(candidates,key=lambda z:z['selection_key']);torch.save({'head':best,'config':config,'preparation':prep},out/'fusion_mlp_qat.pt');report={'status':'PASS','selected':{k:v for k,v in best.items() if k!='state'},'all_candidates':summaries,'seconds':time.time()-begin,'DS2_used':False};(out/'selection_results.json').write_text(json.dumps(report,indent=2));(out/'FROZEN.json').write_text(json.dumps({'status':'PASS','checkpoint':'fusion_mlp_qat.pt','sha256':digest(out/'fusion_mlp_qat.pt'),'DS1_only_selection':True,'previous_DS2_results_known':True,'DS2_not_evaluated_in_followup':True},indent=2));print('FROZEN_NONLINEAR',json.dumps(report['selected']),flush=True)
