from pathlib import Path
import json,time,numpy as np,torch
from train_current import load_checkpoint
from qcare.model import CurrentIRN,calibrate
root=Path(__file__).resolve().parents[1];cache=root/'experiments/cache';data=dict(np.load(cache/'beats.npz'));prep=json.loads((cache/'preparation.json').read_text());maps=np.load(cache/'maps.npy',mmap_mode='r');train=np.flatnonzero(np.isin(data['record'],prep['train_records']));ix=[]
for c in range(5):ix.extend(train[data['label'][train]==c][:2].tolist())
ix=np.array(ix);torch.set_num_threads(1);torch.manual_seed(26)
x=torch.tensor(maps[ix].astype(np.float32))/128;y=torch.tensor(data['label'][ix]);model=CurrentIRN(26,0);ck=load_checkpoint(root/'experiments/ds1_current/best_float.pt',weights_only=True);model.load_state_dict(ck['state_dict']);history=[];start=time.time()
for phase,epochs in [('float',200),('qat',400)]:
 if phase=='qat':model.configure_qat(calibrate(model,maps[ix]));model.clip_latent()
 opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.8)
 for e in range(epochs):
  model.train();opt.zero_grad(set_to_none=True);lg=model(x);loss=torch.nn.functional.cross_entropy(lg,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();model.clip_latent()
  if e%20==0 or e==epochs-1:
   model.eval()
   with torch.no_grad():out=model(x);accuracy=float((out.argmax(1)==y).float().mean());ce=float(torch.nn.functional.cross_entropy(out,y))
   history.append({'phase':phase,'epoch':e,'loss':ce,'accuracy':accuracy});print(history[-1],flush=True)
model.eval()
with torch.no_grad():out=model(x);acc=float((out.argmax(1)==y).float().mean());ce=float(torch.nn.functional.cross_entropy(out,y))
report={'status':'PASS' if acc==1 else 'FAIL','experiment':'memorization of 10 genuine annotated DS1 beats, two per class','synthetic_training':False,'architecture_unchanged':True,'seed':26,'base_model':'DS1 floating warm-start','float_diagnostic_epochs':200,'QAT_diagnostic_epochs':400,'learning_rate':.01,'momentum':.8,'dropout_for_memorization':0,'final_accuracy':acc,'final_cross_entropy':ce,'history':history,'source':[{'record':str(data['record'][i]),'peak':int(data['peak'][i]),'label':int(data['label'][i])} for i in ix],'seconds':time.time()-start,'not_full_DS1_DS2_accuracy':True}
(root/'reports/real_qat_memorization.json').write_text(json.dumps(report,indent=2))
