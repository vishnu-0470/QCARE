from pathlib import Path
import json,numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from qcare.provenance import verify_prepared
root=Path(__file__).resolve().parents[1];cache=root/'experiments/cache'
if (root/'experiments/ds2_evaluation/EVALUATED.json').exists():raise RuntimeError('DS2 already evaluated; preserve the frozen RR baseline')
prep=verify_prepared(root);data=dict(np.load(cache/'beats.npz'));train=np.flatnonzero(np.isin(data['record'],prep['train_records']));val=np.flatnonzero(np.isin(data['record'],prep['validation_records']))
model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,class_weight='balanced',random_state=26))
with threadpool_limits(2):model.fit(data['rr'][train],data['label'][train])
out=root/'experiments/rr_baseline';out.mkdir(parents=True,exist_ok=True);joblib.dump(model,out/'model.joblib')
from qcare.metrics import score
report={'status':'PASS','classifier':'StandardScaler + class-balanced multinomial logistic regression','C':1.,'max_iter':1000,'random_state':26,'features':prep['RR_feature_names'],'context':'causal past RR intervals; no next-RR or future record normalization','fit_records':prep['train_records'],'validation_records':prep['validation_records'],'training_beats':len(train),'DS2_not_evaluated':True,'fit_normalizer_on':'DS1 fit partition only','validation_metrics':score(data['label'][val],model.predict(data['rr'][val]))}
(out/'training.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
