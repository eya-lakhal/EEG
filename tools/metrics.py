import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score
from src.detection import *
from src.data_loader import load_annotations
ds=pd.read_csv('cache/dataset_all19_w2.0_o50_p10.csv'); S=pd.read_csv('cache/loo_scores.csv'); y=ds.Class.values; an=load_annotations()
thr0={"Decision Tree":.5,"Random Forest":.5,"KNN":.5,"SVM":0.0}
for n in S:
    s=S[n].values; p=(s>thr0[n]).astype(int)
    print("%-14s AUC %.3f PR-AUC %.3f Recall %.3f Prec %.3f F1 %.3f"%(n,roc_auc_score(y,s),average_precision_score(y,s),recall_score(y,p),precision_score(y,p),f1_score(y,p)))
pat=[r for r in ds.rec_id.unique() if ds[ds.rec_id==r].Class.sum()>0]; idx={r:np.where(ds.rec_id==r)[0] for r in pat}
rng=np.random.default_rng(0); ap={n:[] for n in S}; f1={n:[] for n in S}
for _ in range(300):
    ii=np.concatenate([idx[r] for r in rng.choice(pat,len(pat))])
    for n in S: ap[n].append(average_precision_score(y[ii],S[n].values[ii])); f1[n].append(f1_score(y[ii],(S[n].values[ii]>thr0[n]).astype(int)))
for n in S: print(n,"PR-AUC %.2f [%.2f-%.2f] F1 %.2f [%.2f-%.2f]"%(np.mean(ap[n]),*np.percentile(ap[n],[2.5,97.5]),np.mean(f1[n]),*np.percentile(f1[n],[2.5,97.5])))
for n in S:
    if n!="Random Forest": print("P(RF>%s) PRAUC %.2f F1 %.2f"%(n,(np.array(ap["Random Forest"])>np.array(ap[n])).mean(),(np.array(f1["Random Forest"])>np.array(f1[n])).mean()))
pp=pd.DataFrame({n:[f1_score(y[idx[r]],(S[n].values[idx[r]]>thr0[n]).astype(int)) for r in pat] for n in S},index=pat)
print("F1 moyen/patient",pp.mean().round(3).to_dict(),"gagnants",pp.idxmax(axis=1).value_counts().to_dict())
for name,thr in [("Random Forest",0.5),("Random Forest",0.3),("SVM",0.0)]:
    H=N=FA=0; ons=[]; low=[]
    for r in ds.rec_id.unique():
        m=(ds.rec_id==r).values; d=ds[m]; s=S[name].values[m]
        iv=predictions_to_intervals(d.t_start,d.t_end,(s>thr).astype(int),smooth=3,min_dur=2.0)
        t=an[an.rec_id==r]; h,n,fa=compare_intervals(t,iv); H+=h;N+=n;FA+=fa
        if n and h<n: low.append((r,h,n))
        for _,z in t.iterrows():
            c=iv[(iv.debut_s<=z.fin_s+2)&(iv.fin_s>=z.debut_s-2)]
            if len(c): ons.append(c.debut_s.iloc[0]-z.debut_s)
    hrs=len(ds)/3600
    print(name,thr,"detected %d/%d FA %d (%.2f/h) onset med %.1f"%(H,N,FA,FA/hrs,np.median(ons)),low)
