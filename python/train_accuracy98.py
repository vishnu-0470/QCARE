"""Strong DS1-only software experiment; not a 4-bit FPGA model claim."""
from pathlib import Path
import json,time,joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from qcare.metrics import score
from qcare.provenance import verify_prepared,digest
from qcare.strong_features import extended_features
r=Path(__file__).resolve().parents[1];out=r/'experiments/accuracy98_software';out.mkdir(exist_ok=True,parents=True)
if (out/'FROZEN.json').exists():raise RuntimeError('Software experiment already frozen')
p=verify_prepared(r);d=dict(np.load(r/'experiments/cache/beats.npz'));g=np.load(r/'experiments/accuracy90_head/DS1_frozen_gap_features.npz');ids=g['indices'];fit=np.isin(d['record'][ids],p['train_records']);y=d['label'][ids];cache=out/'DS1_extended_features.npz'
if cache.exists():X=np.load(cache)['features']
else:X=extended_features(r,d,ids,g['features']);np.savez_compressed(cache,indices=ids,features=X)
configs=[{'max_leaf_nodes':31,'max_depth':8,'min_samples_leaf':30,'l2_regularization':10.,'class_weight_exponent':0.},{'max_leaf_nodes':15,'max_depth':6,'min_samples_leaf':60,'l2_regularization':20.,'class_weight_exponent':.25}]
config={'status':'PASS','method':'software HistGradientBoosting on original frozen GAP16, causal morphology16, past-RR16, normalized real waveform80 and causal template residual80','features':X.shape[1],'candidates':configs,'max_iter':240,'learning_rate':.06,'random_state':526,'fit_records':p['train_records'],'validation_records':p['validation_records'],'DS2_used_for_fit_or_selection':False,'previous_DS2_results_known':True,'selection':'highest DS1 validation accuracy; ties by macroF1','data':'real ECG only; no synthetic ECG','context':'past32 RR and past16 waveform templates; no nextRR, future record normalization or patient ID feature','hardware_status':'software diagnostic only; not exported to 4W/8A RTL','target_accuracy':.98,'baseline_sha256':digest(r/'experiments/ds1_current/best_qat.pt')};(out/'config.json').write_text(json.dumps(config,indent=2));begin=time.time();results=[];best=None;counts=np.bincount(y[fit],minlength=5)
with threadpool_limits(1):
 for k,cfg in enumerate(configs):
  kwargs={k:v for k,v in cfg.items() if k!='class_weight_exponent'};model=HistGradientBoostingClassifier(**kwargs,max_iter=240,learning_rate=.06,early_stopping=False,random_state=526)
  weights=(counts.sum()/counts[y[fit]])**cfg['class_weight_exponent'];model.fit(X[fit],y[fit],sample_weight=weights);m=score(y[~fit],model.predict(X[~fit]));joblib.dump(model,out/f'candidate{k}.joblib');result={'candidate':k,'config':cfg,'metrics':m,'seconds_so_far':time.time()-begin};results.append(result);print(json.dumps(result),flush=True)
  if best is None or (m['accuracy'],m['macro_f1'])>(best['metrics']['accuracy'],best['metrics']['macro_f1']):best=result
joblib.dump(joblib.load(out/f'candidate{best["candidate"]}.joblib'),out/'selected.joblib');(out/'selection_results.json').write_text(json.dumps({'status':'PASS','selected':best,'candidates':results},indent=2));(out/'FROZEN.json').write_text(json.dumps({'status':'PASS','checkpoint':'selected.joblib','sha256':digest(out/'selected.joblib'),'DS1_only_selection':True,'previous_DS2_results_known':True,'DS2_not_evaluated_in_this_experiment':True},indent=2));print('FROZEN_SOFTWARE',best['metrics']['accuracy'],best['metrics']['macro_f1'],flush=True)
