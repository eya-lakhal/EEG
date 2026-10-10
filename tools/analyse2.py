import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score
from src.data_loader import load_annotations
from src.detection import predictions_to_intervals
ds=pd.read_csv('cache/dataset_all19_w2.0_o50_p10.csv'); an=load_annotations(); y=ds.Class.values
A=pd.read_csv('cache/loo_scores.csv'); B=pd.read_csv('cache/loo_scores_recnorm.csv')
thr={"Decision Tree":.5,"Random Forest":.5,"KNN":.5,"SVM":0.0}
pat=[r for r in ds.rec_id.unique() if ds[ds.rec_id==r].Class.sum()>0]; idx={r:np.where(ds.rec_id==r)[0] for r in pat}
print("== Fenêtres (poolé) : baseline -> normalisation par enregistrement")
for n in A:
    for lab,S in (("base",A),("norm",B)):
        s=S[n].values; p=(s>thr[n]).astype(int)
        print("%-14s %s AUC %.3f PR-AUC %.3f R %.3f P %.3f F1 %.3f"%(n,lab,roc_auc_score(y,s),average_precision_score(y,s),recall_score(y,p),precision_score(y,p),f1_score(y,p)))
rng=np.random.default_rng(0); d=[]
for _ in range(300):
    ii=np.concatenate([idx[r] for r in rng.choice(pat,len(pat))]); d.append(average_precision_score(y[ii],B["Random Forest"].values[ii])-average_precision_score(y[ii],A["Random Forest"].values[ii]))
print("RF: gain PR-AUC (norm - base) bootstrap: moy %.3f IC95 [%.3f ; %.3f] ; P(gain>0)=%.2f"%(np.mean(d),*np.percentile(d,[2.5,97.5]),(np.array(d)>0).mean()))
for r in ['230718A_A','220121B_D','210427B_C']:
    g=idx[r]; print(r,"F1 RF base %.2f -> norm %.2f"%(f1_score(y[g],(A['Random Forest'].values[g]>.5).astype(int)),f1_score(y[g],(B['Random Forest'].values[g]>.5).astype(int))))
pp=lambda S,n:np.mean([f1_score(y[idx[r]],(S[n].values[idx[r]]>thr[n]).astype(int)) for r in pat])
print("F1 moyen par patient:",{n:(round(pp(A,n),3),round(pp(B,n),3)) for n in A})

# ---------- détection : bornes + appariement strict
def iou(a,b): i=max(0,min(a[1],b[1])-max(a[0],b[0])); return i/(max(a[1],b[1])-min(a[0],b[0])-0+1e-9) if i>0 else 0
def evaluate(S,n,th,shift=0.0,iou_min=0.3):
    TP=FP=FN=0; es=[]; ee=[]
    for r in ds.rec_id.unique():
        m=(ds.rec_id==r).values; d=ds[m]
        iv=predictions_to_intervals(d.t_start,d.t_end,(S[n].values[m]>th).astype(int),smooth=3,min_dur=2.0)
        P=[(a+shift,b-shift) for a,b in zip(iv.debut_s,iv.fin_s)]; T=list(zip(an[an.rec_id==r].debut_s,an[an.rec_id==r].fin_s))
        pairs=sorted([(iou(p,t),i,j) for i,p in enumerate(P) for j,t in enumerate(T)],reverse=True); up=set(); ut=set()
        for v,i,j in pairs:
            if v>=iou_min and i not in up and j not in ut: up.add(i); ut.add(j); es.append(P[i][0]-T[j][0]); ee.append(P[i][1]-T[j][1])
        TP+=len(ut); FN+=len(T)-len(ut); FP+=len(P)-len(up)
    hrs=len(ds)/3600
    return TP,FN,FP,TP/(TP+FN),TP/max(TP+FP,1),np.median(es),np.median(ee),np.mean(np.abs(es)),np.mean(np.abs(ee)),FP/hrs
print("\n== Détection, appariement STRICT (1 prédit <-> 1 réel, IoU>=0.3). Erreurs de bornes = prédit - réel (s)")
print("%-34s TP FN FP  Recall Préc | début méd/MAE | fin méd/MAE | FA/h"%"")
for lab,S,n,th in (("RF base",A,"Random Forest",.5),("RF base",A,"Random Forest",.5),("RF norm",B,"Random Forest",.5),("SVM base",A,"SVM",0),("SVM norm",B,"SVM",0)):
    for sh in (0.0,1.0):
        r=evaluate(S,n,th,sh); print("%-14s correction ±%.0fs        %2d %2d %3d  %.2f  %.2f | %+.1f / %.1f | %+.1f / %.1f | %.1f"%(lab,sh,*r[:3],r[3],r[4],r[5],r[7],r[6],r[8],r[9]))
