import sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
ds=pd.read_csv('cache/dataset_all19_w2.0_o50_p10.csv'); y=ds.Class.values
feats=['Std_mean','Min_max','Rel_Hz3_mean','Rel_Hz3_max','LineLength_mean','ZeroCross_mean','ZeroCross_max','Power_Hz3_mean','Max_max','DomFreq_mean','Variance_mean','LineLength_max','Rel_Beta_max','Power_Beta_mean','HjorthComplexity_max']
X=ds[feats].values; recs=ds.rec_id.unique()
pat=[r for r in recs if ds[ds.rec_id==r].Class.sum()>0]; idx={r:np.where(ds.rec_id==r)[0] for r in pat}
def score(make, under=None, name=""):
    t=time.time(); s=np.zeros(len(ds)); rng=np.random.default_rng(0)
    for r in recs:
        te=(ds.rec_id==r).values; tr=np.where(~te)[0]
        if under:
            pos=tr[y[tr]==1]; neg=rng.choice(tr[y[tr]==0],min(under*len(pos),(y[tr]==0).sum()),replace=False); tr=np.concatenate([pos,neg])
        m=make().fit(X[tr],y[tr]); s[te]=m.predict_proba(X[te])[:,1]
    return s
def rep(name,s,ths=(.5,)):
    f1p=np.mean([f1_score(y[idx[r]],(s[idx[r]]>.5).astype(int)) for r in pat])
    line="%-34s PR-AUC %.3f"%(name,average_precision_score(y,s))
    for th in ths:
        p=(s>th).astype(int); line+=" | th %.2f R %.2f P %.2f F1 %.3f"%(th,recall_score(y,p),precision_score(y,p),f1_score(y,p))
    print(line+" | F1/patient(.5) %.3f"%f1p,flush=True)
RF=lambda **k:(lambda:make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=150,n_jobs=-1,random_state=0,**k)))
print("--- Random Forest (150 arbres) : référence = 200 arbres, min_samples_leaf=2, balanced : PR-AUC 0.592, F1 0.641, F1/patient 0.664")
for nm,k in [("msl=2 balanced (réf, 150 arbres)",dict(min_samples_leaf=2,class_weight="balanced")),
             ("msl=5 balanced",dict(min_samples_leaf=5,class_weight="balanced")),
             ("max_features=0.5, msl=2",dict(min_samples_leaf=2,max_features=0.5,class_weight="balanced")),
             ("balanced_subsample, msl=3, depth=12",dict(min_samples_leaf=3,max_depth=12,class_weight="balanced_subsample")),
             ("max_features=0.5, msl=5",dict(min_samples_leaf=5,max_features=0.5,class_weight="balanced"))]:
    rep("RF "+nm,score(RF(**k)),ths=(.5,.35))
print("--- KNN (référence k=7 distance : PR-AUC 0.496, F1 0.565)")
for k in (7,15,31):
    kn=lambda k=k:make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=k,weights="distance",n_jobs=-1))
    rep(f"KNN k={k}",score(kn),ths=(.5,.3,.2))
    rep(f"KNN k={k} + sous-échant. neg x5",score(kn,under=5),ths=(.5,.3,.2))
