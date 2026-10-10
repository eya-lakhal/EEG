import sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.data_loader import load_annotations
from src.detection import predictions_to_intervals
ds=pd.read_csv('cache/dataset_all19_w2.0_o50_p10.csv'); an=load_annotations(); y=ds.Class.values
feats=['Std_mean','Min_max','Rel_Hz3_mean','Rel_Hz3_max','LineLength_mean','ZeroCross_mean','ZeroCross_max','Power_Hz3_mean','Max_max','DomFreq_mean','Variance_mean','LineLength_max','Rel_Beta_max','Power_Beta_mean','HjorthComplexity_max']
# --- normalisation des features PAR enregistrement (sans utiliser les étiquettes)
Xn=ds[feats].copy()
for r,g in ds.groupby('rec_id'):
    med=g[feats].median(); iqr=(g[feats].quantile(.75)-g[feats].quantile(.25)).replace(0,1e-9)
    Xn.loc[g.index]=(g[feats]-med)/iqr
mk={"Decision Tree":lambda:make_pipeline(StandardScaler(),DecisionTreeClassifier(max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=0)),
"Random Forest":lambda:make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=0)),
"KNN":lambda:make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7,weights="distance",n_jobs=-1)),
"SVM":lambda:make_pipeline(StandardScaler(),SVC(kernel="rbf",class_weight="balanced",random_state=0))}
out={}
for n,f in mk.items():
    t=time.time(); s=np.zeros(len(ds))
    for r in ds.rec_id.unique():
        te=(ds.rec_id==r).values; m=f().fit(Xn[~te],y[~te])
        s[te]=m.predict_proba(Xn[te])[:,1] if n!="SVM" else m.decision_function(Xn[te])
    out[n]=s; print(n,round(time.time()-t),'s',flush=True)
pd.DataFrame(out).to_csv('cache/loo_scores_recnorm.csv',index=False)
