from pathlib import Path
import sys,time,json,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_current import load_checkpoint
from qcare.model import CurrentIRN,calibrate
r=Path(__file__).resolve().parents[1];ck=load_checkpoint(r/'experiments/ds1_current/best_float.pt',weights_only=True);m=CurrentIRN();m.load_state_dict(ck['state_dict']);data=dict(np.load(r/'experiments/cache/beats.npz'));maps=np.load(r/'experiments/cache/maps.npy',mmap_mode='r');ix=np.flatnonzero(np.isin(data['record'],ck['config']['train_records']))[:200];x=torch.tensor(maps[ix].astype(np.float32))/128;y=torch.tensor(data['label'][ix]);torch.set_num_threads(2);E=calibrate(m,maps[ix[:64]]);m.configure_qat(E);m.clip_latent();m.train();out={}
for threads in [1,2,4]:
 torch.set_num_threads(threads);t=time.time()
 for step in range(8):
  m.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(m(x),y).backward()
 out[threads]=(time.time()-t)/8
print(json.dumps(out));(r/'reports/cpu_training_benchmark.json').write_text(json.dumps({'status':'PASS','seconds_per_real_DS1_batch_200_forward_backward':out,'no_optimizer_updates':True},indent=2))
