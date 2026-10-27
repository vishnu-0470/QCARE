import numpy as np
from .data import CLASSES

def score(y,pred):
 y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int);cm=np.zeros((5,5),dtype=np.int64);np.add.at(cm,(y,pred),1)
 tp=cm.diagonal();support=cm.sum(1);pp=cm.sum(0);fp=pp-tp;fn=support-tp;tn=cm.sum()-tp-fp-fn
 div=lambda a,b:np.divide(a,b,out=np.zeros(5,dtype=float),where=b!=0)
 pr=div(tp,pp);se=div(tp,support);f1=div(2*pr*se,pr+se);sp=div(tn,tn+fp)
 return {'class_order':CLASSES,'samples':len(y),'accuracy':float(tp.sum()/len(y)),'macro_f1':float(f1.mean()),'macro_f1_present':float(f1[support>0].mean()),'confusion_matrix':cm.tolist(),'per_class':{c:{'support':int(support[i]),'sensitivity':float(se[i]),'positive_predictivity':float(pr[i]),'precision':float(pr[i]),'specificity':float(sp[i]),'f1':float(f1[i])} for i,c in enumerate(CLASSES)}}

def cluster_bootstrap(y,p,records,seed=26,n=1000):
 records=np.asarray(records);unique=np.unique(records);groups={r:np.flatnonzero(records==r) for r in unique};rng=np.random.default_rng(seed);scores=[]
 # Treat 201/202 as one cluster when both occur in a reported partition.
 for _ in range(n):
  ix=np.concatenate([groups[r] for r in rng.choice(unique,len(unique),True)]);m=score(y[ix],p[ix]);scores.append([m['accuracy'],m['macro_f1']])
 q=np.quantile(scores,[.025,.975],axis=0)
 return {'unit':'record_cluster','draws':n,'accuracy_95_ci':q[:,0].tolist(),'macro_f1_95_ci':q[:,1].tolist(),'note':'DS2 contains 202 only; 201 is a conventional DS1 shared-subject caveat, not a duplicate within DS2'}
