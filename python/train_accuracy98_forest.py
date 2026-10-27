"""Final DS1-only forest comparison; separate from the quantized FPGA network."""
from pathlib import Path
import json,time,joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from qcare.metrics import score
from qcare.provenance import digest
r=Path(__file__).resolve().parents[1];out=r/'experiments/accuracy98_forest';out.mkdir(exist_ok=True)
if (out/'FROZEN.json').exists():raise RuntimeError('Forest comparison frozen')
p=json.loads((r/'experiments/cache/preparation.json').read_text());d=np.load(r/'experiments/cache/beats.npz');z=np.load(r/'experiments/accuracy98_software/DS1_extended_features.npz');X=z['features'].astype(np.float32);ids=z['indices'];fit=np.isin(d['record'][ids],p['train_records']);y=d['label'][ids];counts=np.bincount(y[fit],minlength=5);models=[ExtraTreesClassifier(n_estimators=160,max_features=.5,max_depth=24,min_samples_leaf=3,n_jobs=2,random_state=626),RandomForestClassifier(n_estimators=160,max_features='sqrt',max_depth=24,min_samples_leaf=2,n_jobs=2,random_state=626)];results=[];begin=time.time();best=None
(out/'config.json').write_text(json.dumps({'status':'PASS','models':[str(m) for m in models],'fit_records':p['train_records'],'validation_records':p['validation_records'],'DS2_used_for_fit_or_selection':False,'sample_weight_exponent':.15,'selection':'DS1 validation accuracy, ties macroF1','software_only':True,'real_ECG_only':True,'previous_DS2_results_known':True},indent=2))
for k,model in enumerate(models):
 model.fit(X[fit],y[fit],sample_weight=(counts.sum()/counts[y[fit]])**.15);metrics=score(y[~fit],model.predict(X[~fit]));joblib.dump(model,out/f'candidate{k}.joblib',compress=3);entry={'candidate':k,'type':type(model).__name__,'metrics':metrics,'seconds':time.time()-begin};results.append(entry);print(json.dumps(entry),flush=True)
 if best is None or (metrics['accuracy'],metrics['macro_f1'])>(best['metrics']['accuracy'],best['metrics']['macro_f1']):best=entry
(out/'selection_results.json').write_text(json.dumps({'status':'PASS','selected':best,'candidates':results},indent=2));(out/'FROZEN.json').write_text(json.dumps({'status':'PASS','checkpoint':f'candidate{best["candidate"]}.joblib','sha256':digest(out/f'candidate{best["candidate"]}.joblib'),'DS1_only_selection':True,'previous_DS2_results_known':True},indent=2))
