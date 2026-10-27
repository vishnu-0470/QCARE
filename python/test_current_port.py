from pathlib import Path
import json,numpy as np,torch
from qcare.model import CurrentIRN,calibrate
from qcare.program import export_model,run_program
from qcare.frontend import cwt_maps
root=Path(__file__).resolve().parents[1];cache=root/'experiments/cache';data=dict(np.load(cache/'beats.npz'));prep=json.loads((cache/'preparation.json').read_text());maps=np.load(cache/'maps.npy',mmap_mode='r')
torch.set_num_threads(4);train=np.flatnonzero(np.isin(data['record'],prep['train_records']));chosen=[]
for c in range(5):chosen.extend(train[data['label'][train]==c][:2].tolist())
model=CurrentIRN(26,0.5);E=calibrate(model,maps[np.array(chosen)]);model.configure_qat(E);model.clip_latent();model.eval()
ops,w,b=export_model(model,root/'experiments/port_check')
checks=[]
with torch.no_grad():
 for i in chosen:
  x=maps[i];logits,trace=model(torch.tensor(x.astype(np.float32))[None]/128,True)
  acc=np.rint(logits[0].numpy()/2.0**(E['blk4']+model.WE['fc'])).astype(np.int64)
  oracle,intermediates=run_program(x,ops,w,b,True);np.testing.assert_array_equal(acc,oracle)
  tensor_names=['ds1','irn1_A','irn1_BM','irn1_B1P','irn1','irn2_A','irn2_BM','irn2_B1P','irn2','pool','ds2','irn3_A','irn3_BM','irn3_B1P','irn3','irn4_A','irn4_BM','irn4_B1P','irn4','gap','logits']
  for name,value in zip(tensor_names,intermediates):
   exp=E['ds1'] if name=='ds1' else E['ds2'] if name=='ds2' else E['blk2'] if name=='pool' else E['blk4'] if name=='gap' else E['blk4']+model.WE['fc'] if name=='logits' else E['blk'+name[3]]
   q=np.rint(trace[name][0].numpy()/2.0**exp).astype(np.int64)
   np.testing.assert_array_equal(q.reshape(value.shape),value)
  checks.append({'record':str(data['record'][i]),'peak':int(data['peak'][i]),'label':int(data['label'][i])})
report={'status':'PASS','architecture_parameter_count':sum(p.numel() for p in model.parameters()),'real_DS1_beats_checked':checks,'all_21_operation_outputs_bit_exact':True,'golden_scope':'PyTorch QAT vs independent NumPy program oracle; no DS2 used','exponents':E}
(root/'reports/current_port_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
