"""Run the DS1-selected software classifier on an annotated real ECG beat."""
from pathlib import Path
import argparse,json,joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from qcare.record_input import load_record_data
from qcare.strong_features import extended_features
from qcare.frontend import cwt_maps,quantize_map
from qcare.model import CurrentIRN
from qcare.data import CLASSES
from qcare.provenance import check_file
from train_current import load_checkpoint
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--record',default='100');p.add_argument('--beat-index',type=int,default=0);p.add_argument('--output');a=p.parse_args();selection=json.loads((r/'experiments/accuracy98_selected/FROZEN.json').read_text())['selected'];model_path=r/'experiments'/selection['folder']/selection['checkpoint'];check_file(model_path,selection['sha256']);classifier=joblib.load(model_path);signal,data=load_record_data(r,a.record)
if not 0<=a.beat_index<len(data['peak']):raise ValueError('Beat index out of range')
n=a.beat_index+1;ids=np.arange(n);beats=np.stack([signal[v-140:v+220] for v in data['peak'][:n]]);maps=quantize_map(cwt_maps(beats));ck=load_checkpoint(r/'experiments/ds1_current/best_qat.pt',weights_only=True,map_location='cpu');backbone=CurrentIRN();backbone.load_state_dict(ck['state_dict']);backbone.configure_qat(ck['E'],ck['WE']);backbone.double().eval();torch.set_num_threads(1);g=[]
with torch.no_grad():
 for j in range(0,n,200):
  _,trace=backbone(torch.tensor(maps[j:j+200].astype(np.float64))/128,True);g.append(trace['gap'].numpy())
features=extended_features(r,data,ids,np.concatenate(g))
with threadpool_limits(1):pred=int(classifier.predict(features[-1:])[0]);prob=classifier.predict_proba(features[-1:])[0]
out=r/(a.output or f'experiments/software_replay/record{a.record}_beat{a.beat_index}');out.mkdir(exist_ok=True,parents=True);result={'status':'PASS','model':'software classifier; not the 4W FPGA network','record':a.record,'beat_index':a.beat_index,'R_sample_0based':int(data['peak'][a.beat_index]),'annotated_class':CLASSES[int(data['label'][a.beat_index])],'predicted_class':CLASSES[pred],'software_scores':dict(zip(CLASSES,map(float,prob))),'model_sha256':selection['sha256'],'output_directory':str(out)};(out/'inference.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
