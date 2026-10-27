"""Evaluate only the DS1-selected software candidate; keep FPGA results separate."""
from pathlib import Path
import json,time,csv,joblib,numpy as np
from threadpoolctl import threadpool_limits
from qcare.provenance import verify_prepared,digest,check_file
from qcare.strong_features import extended_features
from qcare.metrics import score,cluster_bootstrap
r=Path(__file__).resolve().parents[1];out=r/'experiments/accuracy98_selected';out.mkdir(exist_ok=True)
if (out/'EVALUATED.json').exists():raise RuntimeError('Software DS2 follow-up already evaluated')
p=verify_prepared(r);candidates=[]
for folder in ['accuracy98_software','accuracy98_forest']:
 run=r/'experiments'/folder;f=json.loads((run/'FROZEN.json').read_text());check_file(run/f['checkpoint'],f['sha256']);z=json.loads((run/'selection_results.json').read_text())['selected'];candidates.append({'folder':folder,'checkpoint':f['checkpoint'],'sha256':f['sha256'],'validation_accuracy':z['metrics']['accuracy'],'validation_macro_f1':z['metrics']['macro_f1']})
best=max(candidates,key=lambda v:(v['validation_accuracy'],v['validation_macro_f1']));(out/'FROZEN.json').write_text(json.dumps({'status':'PASS','selected':best,'DS1_candidates':candidates,'selection':'DS1 validation accuracy, ties macroF1','previous_DS2_results_known':True,'no_followup_DS2_model_selection':True},indent=2))
model=joblib.load(r/'experiments'/best['folder']/best['checkpoint']);d=dict(np.load(r/'experiments/cache/beats.npz'));test=np.flatnonzero(np.isin(d['record'],p['DS2']));z=np.load(r/'experiments/accuracy90_fusion_mlp/DS2_followup/predictions.npz');np.testing.assert_array_equal(test,z['indices']);gap=z['gap_codes']*2.**(-3);begin=time.time();X=extended_features(r,d,test,gap)
with threadpool_limits(1):pred=model.predict(X);prob=model.predict_proba(X)
labels=d['label'][test];m=score(labels,pred);m.update(status='PASS',meaning='evaluation executed; target separately assessed',target_accuracy=.98,target_status='PASS' if m['accuracy']>.98 else 'FAIL',model_family='software HistGradientBoosting/forest selection; not a quantized FPGA network',selected_model=best,DS2_records=p['DS2'],fit_records=p['train_records'],validation_records=p['validation_records'],test_beats=len(test),boundary_exclusions=24,previous_DS2_results_known=True,DS2_used_for_followup_fitting_or_selection=False,uncertainty=cluster_bootstrap(labels,pred,d['record'][test]),seconds=time.time()-begin)
np.savez_compressed(out/'predictions.npz',indices=test,record=d['record'][test],labels=labels,predictions=pred,probabilities=prob)
with open(out/'predictions.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['record','R_sample_0based','true_class','predicted_class'])
 for j,i in enumerate(test):w.writerow([d['record'][i],int(d['peak'][i]),int(labels[j]),int(pred[j])])
(out/'metrics.json').write_text(json.dumps(m,indent=2));(r/'reports/accuracy98/DS2_metrics.json').write_text(json.dumps(m,indent=2));(out/'EVALUATED.json').write_text(json.dumps({'status':'PASS','accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'target_status':m['target_status'],'selected_model_sha256':best['sha256'],'selection_completed_before_this_evaluation':True,'previous_DS2_results_known':True},indent=2));print(json.dumps({k:m[k] for k in ['accuracy','macro_f1','target_status','per_class']},indent=2))
