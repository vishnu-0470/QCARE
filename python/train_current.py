"""Full current IRN training; DS2 arrays are never indexed here."""
from pathlib import Path
import argparse,json,time,hashlib,copy,random,sys
import numpy as np
import torch
from qcare.model import CurrentIRN,calibrate
from qcare.metrics import score
from qcare.data import sha
from qcare.provenance import verify_prepared,verify_checkpoint_preparation

def load_checkpoint(path,**kwargs):
 # Earlier local checkpoints contain TorchVersion, a string subclass. Allow
 # only that known metadata class while retaining weights_only safety.
 with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
  return torch.load(path,**kwargs)

def predict(model,maps,indices,batch=200):
 model.eval();pred=[]
 with torch.no_grad():
  for start in range(0,len(indices),batch):
   x=torch.from_numpy(np.asarray(maps[indices[start:start+batch]]).astype(np.float32))/128
   pred.extend(model(x).argmax(1).tolist())
 return np.array(pred)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='experiments/ds1_current');p.add_argument('--epochs',type=int,default=300)
 p.add_argument('--float-epochs',type=int,default=30);p.add_argument('--seed',type=int,default=26);p.add_argument('--threads',type=int,default=4)
 p.add_argument('--resume');p.add_argument('--stop-after',type=int);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];cache=root/'experiments/cache';out=root/a.run;out.mkdir(exist_ok=True,parents=True)
 prep=verify_prepared(root);data=dict(np.load(cache/'beats.npz'))
 maps=np.load(cache/'maps.npy',mmap_mode='r');aug=np.load(cache/'augmented_maps.npy',mmap_mode='r');lookup=np.load(cache/'aug_lookup.npy')
 train=np.flatnonzero(np.isin(data['record'],prep['train_records']));val=np.flatnonzero(np.isin(data['record'],prep['validation_records']))
 assert not set(data['record'][train])&set(prep['DS2']);assert not set(data['record'][val])&set(prep['DS2'])
 if (out/'FROZEN.json').exists():raise RuntimeError('Frozen run already exists; use a new --run directory')
 if (out/'last.pt').exists() and not a.resume:raise RuntimeError('Run exists; explicitly resume it or choose a new --run directory')
 config={'architecture':'unchanged current ds1/IRN1/IRN2/pool/ds2/IRN3/IRN4/GAP/FC','seed':a.seed,'epochs':a.epochs,'float_warmup_epochs':a.float_epochs,'qat_epochs':a.epochs-a.float_epochs,'optimizer':'SGD','momentum':.8,'initial_learning_rate':.01,'batch_size':200,'dropout':.5,'loss':'cross_entropy','schedule':'cosine LR per phase','weight_decay':0.,'balanced_epoch_sampling':{'N':5000,'V':'min(5000,available)','S':2000,'F':2000,'Q':2000},'augmentation':'real waveform shift b-60, b integer uniform 40..80','train_records':prep['train_records'],'validation_records':prep['validation_records'],'DS2_used_for_fit_selection_calibration':False,'class_order':['N','S','V','F','Q'],'input_contract':prep['cwt'],'dataset_manifest_sha256':prep['manifest_sha256'],'cache_sha256':prep['cache_files'],'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'threads':a.threads,'command':' '.join(sys.argv),'paper_deviations':['current archived architecture has 14 convolutions/four IRN blocks, not the paper six-layer model','30 float warm-up + 270 QAT epochs by default; paper QAT warm-start schedule unavailable','cosine schedule chosen because full LR schedule is not stated','cap V selection if fewer than 5000 DS1 training beats','B16 accumulator-scale biases retained to match current RTL','holdout 4 DS1 records for patient-aware validation']}
 (out/'config.json').write_text(json.dumps(config,indent=2))
 torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True);torch.manual_seed(a.seed);random.seed(a.seed);rng=np.random.default_rng(a.seed)
 model=CurrentIRN(a.seed,.5);optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.8)
 history=[];best_float=-1.;best_qat=-1.;start_ep=0;start_time=time.time()
 if a.resume:
  ck=load_checkpoint(root/a.resume,weights_only=True,map_location='cpu');verify_checkpoint_preparation(ck,prep);model.load_state_dict(ck['state_dict']);model.qat=ck['qat'];model.E=ck['E'];model.WE=ck['WE'];optimizer.load_state_dict(ck['optimizer']);history=ck['history'];best_float=ck['best_float'];best_qat=ck['best_qat'];start_ep=ck['epoch']+1;rng.bit_generator.state=json.loads(ck['numpy_rng_json']);torch.set_rng_state(ck['torch_rng'])
 for epoch in range(start_ep,a.epochs):
  if epoch==a.float_epochs:
   ck=load_checkpoint(out/'best_float.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state_dict'])
   # Calibrate once on a fixed, class-balanced DS1 TRAINING-only subset.
   chosen=[]
   for c in range(5):
    ix=train[data['label'][train]==c]
    if len(ix):chosen.extend(rng.choice(ix,min(128,len(ix)),replace=False).tolist())
   E=calibrate(model,maps[np.array(chosen)]);model.configure_qat(E);model.clip_latent();optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.8)
   (out/'calibration.json').write_text(json.dumps({'status':'PASS','train_indices':chosen,'records':sorted(set(data['record'][chosen])),'activation_exponents':E,'frozen_weight_exponents':model.WE},indent=2))
  phase='float' if epoch<a.float_epochs else 'qat';phase_epoch=epoch if phase=='float' else epoch-a.float_epochs
  phase_length=a.float_epochs if phase=='float' else a.epochs-a.float_epochs
  lr=.0001+(.01-.0001)*.5*(1+np.cos(np.pi*phase_epoch/max(1,phase_length-1)))
  for group in optimizer.param_groups:group['lr']=float(lr)
  indices=[]
  for c in range(5):
   ix=train[data['label'][train]==c]
   if not len(ix):raise ValueError('Training class missing: '+str(c))
   n=min(5000,len(ix)) if c in (0,2) else 2000
   indices.extend(rng.choice(ix,n,replace=(n>len(ix))).tolist())
  rng.shuffle(indices);model.train();total_loss=0.;right=0;count=0;begin=time.time()
  for start in range(0,len(indices),200):
   ix=np.array(indices[start:start+200]);x=np.array(maps[ix],copy=True)
   rare=lookup[ix]>=0
   if rare.any():x[rare]=aug[lookup[ix[rare]],rng.integers(0,41,rare.sum())]
   x=torch.from_numpy(x.astype(np.float32))/128;y=torch.from_numpy(data['label'][ix].astype(np.int64))
   optimizer.zero_grad(set_to_none=True);lg=model(x);loss=torch.nn.functional.cross_entropy(lg,y)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();model.clip_latent()
   total_loss+=float(loss.detach())*len(ix);right+=int((lg.argmax(1)==y).sum());count+=len(ix)
  predictions=predict(model,maps,val);m=score(data['label'][val],predictions)
  line={'epoch':epoch,'phase':phase,'learning_rate':float(lr),'train_loss':total_loss/count,'train_accuracy':right/count,'validation_accuracy':m['accuracy'],'validation_macro_f1':m['macro_f1'],'seconds':time.time()-begin,'total_seconds':time.time()-start_time};history.append(line)
  state={'state_dict':model.state_dict(),'qat':model.qat,'E':model.E,'WE':model.WE,'epoch':epoch,'config':config,'preparation':prep}
  if phase=='float' and m['macro_f1']>best_float:best_float=m['macro_f1'];torch.save(state,out/'best_float.pt')
  if phase=='qat' and m['macro_f1']>best_qat:best_qat=m['macro_f1'];torch.save(state,out/'best_qat.pt')
  torch.save({**state,'optimizer':optimizer.state_dict(),'history':history,'best_float':best_float,'best_qat':best_qat,'numpy_rng_json':json.dumps(rng.bit_generator.state),'torch_rng':torch.get_rng_state()},out/'last.pt')
  (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(line),flush=True)
  if a.stop_after is not None and epoch+1>=a.stop_after:break
 if len(history)==a.epochs:
  best=out/'best_qat.pt'
  (out/'FROZEN.json').write_text(json.dumps({'status':'PASS','checkpoint':'best_qat.pt','checkpoint_sha256':sha(best),'selection':'DS1 patient-held-out validation macro-F1 only','DS2_not_evaluated_by_trainer':True,'completed_epochs':a.epochs,'frozen_before_DS2_evaluation':True},indent=2))
if __name__=='__main__':main()
