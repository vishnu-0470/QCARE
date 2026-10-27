"""DS1 validation diagnostics only; no DS2 inference."""
from pathlib import Path
import json,numpy as np,joblib,time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from qcare.metrics import score
r=Path(__file__).resolve().parents[1];p=json.loads((r/'experiments/cache/preparation.json').read_text());d=np.load(r/'experiments/cache/beats.npz');g=np.load(r/'experiments/accuracy90_head/DS1_frozen_gap_features.npz');a=np.load(r/'experiments/accuracy90_fusion/DS1_aux_features.npz');np.testing.assert_array_equal(g['indices'],a['indices']);ids=g['indices'];fit=np.isin(d['record'][ids],p['train_records']);X=np.c_[g['features'],a['features']];y=d['label'][ids];out=r/'experiments/accuracy90_diagnostics';out.mkdir(exist_ok=True);reports={}
with threadpool_limits(1):
 for C in [.001,.01,.1,1.]:
  model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000,random_state=326));model.fit(X[fit],y[fit]);m=score(y[~fit],model.predict(X[~fit]));reports['logreg_C'+str(C)]=m;joblib.dump(model,out/('logreg_C'+str(C)+'.joblib'));print(C,m['accuracy'],m['macro_f1'],flush=True)
 model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,max_depth=5,min_samples_leaf=40,learning_rate=.08,l2_regularization=10.,early_stopping=False,random_state=326)
 model.fit(X[fit],y[fit]);m=score(y[~fit],model.predict(X[~fit]));reports['HGBT_diagnostic_only']=m;joblib.dump(model,out/'teacher.joblib');print('HGBT',m['accuracy'],m['macro_f1'],flush=True)
(out/'DS1_validation_only.json').write_text(json.dumps({'status':'PASS','DS2_evaluated':False,'scope':'software diagnostics, not claimed as FPGA network results','metrics':reports},indent=2))
