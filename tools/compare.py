import warnings; warnings.filterwarnings("ignore")
import sys, time, numpy as np, pandas as pd
sys.path.insert(0,'.')
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, recall_score, precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.dataset import feature_cols
from src.selection import rank_features, select_uncorrelated
ds=pd.read_csv('cache/dataset_all19_w2.0_o50_p10.csv'); F=[c for c in feature_cols(ds) if c!='Class3']
# NB: le classement des features est fait sur tout le dataset (légère fuite, identique pour les 4 modèles)
rk=rank_features(ds,F); feats=select_uncorrelated(rk,ds,15,0.9); print(feats)
mk={"Decision Tree":lambda:make_pipeline(StandardScaler(),DecisionTreeClassifier(max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=0)),
"Random Forest":lambda:make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=0)),
"KNN":lambda:make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7,weights="distance",n_jobs=-1)),
"SVM":lambda:make_pipeline(StandardScaler(),SVC(kernel="rbf",class_weight="balanced",random_state=0))}
recs=ds.rec_id.unique(); y=ds.Class.values; out={}
for n,f in mk.items():
    t=time.time(); s=np.zeros(len(ds))
    for r in recs:
        te=(ds.rec_id==r).values; m=f().fit(ds.loc[~te,feats],y[~te])
        s[te]=m.predict_proba(ds.loc[te,feats])[:,1] if n!="SVM" else m.decision_function(ds.loc[te,feats])
    out[n]=s; print(n,round(time.time()-t),'s',flush=True)
pd.DataFrame(out).to_csv('cache/loo_scores.csv',index=False)
