#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulation framework **v12** – repeat until performance criterion met & save on success
====================================================================================
* 1) 독립 경로 중 무작위 20개를 선택
* 2) 이 20개 중 무작위 10개를 causal (true signal) 지정하여 β∼N(0,5),
     나머지 10개는 β=0 (false signal)
* 3) α∼N(0,5) for all genes in the 20-pathway pool
* 4) linear & nonlinear 관계별 Y 생성
* 5) L1-penalty 로지스틱 회귀 학습하여 Test AUC ≥ `--auc_thr` 이면
     해당 시나리오 데이터를 `save_all()`로 저장하고 종료
* 6) **--seed_base** 와 **--seed_split** 로 완전 재현 가능
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def load_mutation(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, index_col=0)
    print(f"Loaded mutation data: {df.shape[0]}×{df.shape[1]}")
    return df

def load_gmt(fp: Path) -> Dict[str, set]:
    pathways: Dict[str, set] = {}
    with fp.open() as fh:
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts)>=3:
                pathways[parts[0]] = set(parts[2:])
    print(f"Loaded {len(pathways)} pathways")
    return pathways

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def remove_outliers(df: pd.DataFrame, z_thresh: float) -> pd.DataFrame:
    if z_thresh<=0: return df
    pcs = PCA(n_components=2).fit_transform(df.values)
    z = np.abs(stats.zscore(pcs,axis=0))
    mask=(z>z_thresh).any(axis=1)
    if mask.any(): print(f"Removed {mask.sum()} outliers")
    return df.loc[~mask]

def calibrate_intercept(eta: np.ndarray, prev: float) -> float:
    lo,hi=-20,20
    for _ in range(40):
        mid=(lo+hi)/2
        if expit(eta+mid).mean()>prev: hi=mid
        else: lo=mid
    return mid

def jaccard(a:set,b:set)->float:
    return len(a&b)/len(a|b) if (a or b) else 1.0

def independent_paths(all_paths:Dict[str,set],thr:float=0.20)->List[str]:
    keep=[]
    for p in all_paths:
        sims=[jaccard(all_paths[p],all_paths[o]) for o in all_paths if o!=p]
        if sims and np.mean(sims)<thr: keep.append(p)
    return keep

def _nonlinear(x:np.ndarray)->np.ndarray:
    return np.tanh(x)

def evaluate_auc(X:pd.DataFrame, y:pd.Series, splits:dict[str,List[int]])->Dict[str,float]:
    model=LogisticRegression(penalty="l1",solver="liblinear",max_iter=2000)
    model.fit(X.loc[splits["train"]], y.loc[splits["train"]])
    aucs={}
    for sp in ("train","val","test"):
        prob=model.predict_proba(X.loc[splits[sp]])[:,1]
        aucs[sp]=roc_auc_score(y.loc[splits[sp]],prob)
    print(f"AUCs – train:{aucs['train']:.3f}, val:{aucs['val']:.3f}, test:{aucs['test']:.3f}")
    return aucs

# ---------------------------------------------------------------------
# Data generation (no disk I/O)
# ---------------------------------------------------------------------
def trial_once(
    rel:Relationship, mut_df:pd.DataFrame, paths:Dict[str,set],
    prev:float, z_thresh:float, seed: int, seed_split: int
)->Tuple[pd.DataFrame,pd.Series,dict[str,List[int]],List[str],List[str],Dict[str,float]]:
    rng=np.random.RandomState(seed)
    indep=independent_paths(paths)
    pool20=rng.choice(indep,20,replace=False).tolist()
    true10=rng.choice(pool20,10,replace=False).tolist()
    false10=[p for p in pool20 if p not in true10]

    genes=set().union(*(paths[p] for p in pool20))
    X=mut_df[list(genes & set(mut_df.columns))].copy()
    X=remove_outliers(X,z_thresh)

    alpha={g:rng.normal(0,5) for g in X.columns}
    beta ={p:(rng.normal(0,5) if p in true10 else 0.0) for p in pool20}

    eta=np.zeros(X.shape[0])
    for p in pool20:
        gs=list(set(X.columns)&paths[p])
        if not gs: continue
        sig=X[gs].values @ np.array([alpha[g] for g in gs])
        if rel=="nonlinear": sig=_nonlinear(sig)
        eta+=beta[p]*sig

    c=calibrate_intercept(eta,prev)
    y=pd.Series(rng.binomial(1,expit(eta+c)),index=X.index,name="response")

    ids=X.index
    X_tmp,X_test,y_tmp,y_test=train_test_split(ids,y,stratify=y,test_size=0.10,random_state=seed_split)
    X_tr,X_val,y_tr,y_val=train_test_split(X_tmp,y_tmp,stratify=y_tmp,test_size=0.1111,random_state=seed_split)
    splits={"train":list(X_tr),"val":list(X_val),"test":list(X_test)}

    return X,y,splits,true10,false10,{"eta":eta,"c":c}

# ---------------------------------------------------------------------
# Full save (v8 style)
# ---------------------------------------------------------------------
def save_all(
    outdir:Path, X:pd.DataFrame, y:pd.Series, splits:dict[str,List[int]],
    alpha:Dict[str,float], beta:Dict[str,float], true10:List[str], false10:List[str],
    rel:Relationship, seed:int, seed_split:int
):
    # write seeds
    outdir.mkdir(parents=True,exist_ok=True)
    with open(outdir/"seeds.txt","w") as f:
        f.write(f"seed_base_run: {seed}\nseed_split: {seed_split}\n")

    # PCA plot
    pcs=PCA(n_components=2).fit_transform(X.values)
    plt.figure(figsize=(6,5))
    pal=sns.color_palette("Set1",2)
    for cls in (0,1):
        idx=y==cls
        plt.scatter(pcs[idx,0],pcs[idx,1],s=20,alpha=0.6,edgecolor="k",c=[pal[cls]],label=f"M={cls}")
    plt.title(f"PCA – {rel}, seed={seed}")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(frameon=False)
    plt.savefig(outdir/"pca_plot.png",dpi=200)
    plt.close()

    X.to_csv(outdir/"somatic_mutation_paper.csv")
    y.to_frame().to_csv(outdir/"response.csv",index=True)
    pd.DataFrame(alpha.items(),columns=["gene","alpha"]).to_csv(outdir/"gene_alpha.csv",index=False)
    pd.DataFrame(beta.items(),columns=["pathway","beta"]).to_csv(outdir/"pathway_beta.csv",index=False)
    pd.Series(X.columns,name="genes").to_csv(outdir/"selected_genes.csv",index=False)

    sp=outdir/"splits"; sp.mkdir(exist_ok=True)
    pd.DataFrame({"id":splits["train"],"response":y.loc[splits["train"]]}).to_csv(sp/"training_set_0.csv",index=True)
    pd.DataFrame({"id":splits["val"],"response":y.loc[splits["val"]]}).to_csv(sp/"validation_set.csv",index=True)
    pd.DataFrame({"id":splits["test"],"response":y.loc[splits["test"]]}).to_csv(sp/"test_set.csv",index=True)

# ---------------------------------------------------------------------
# Repeat until success
# ---------------------------------------------------------------------
def run_until_success(
    rels:List[Relationship], mut_df:pd.DataFrame, paths:Dict[str,set],
    prev:float, z_thresh:float, auc_thr:float, max_iter:int,
    seed_base:int, seed_split:int, mut_fp:Path, gmt_fp:Path
):
    for attempt in range(1, max_iter+1):
        seed=seed_base+attempt
        print(f"\n--- Attempt {attempt} (seed={seed}) ---")
        for rel in rels:
            X,y,splits,true10,false10,extra = trial_once(
                rel, mut_df, paths, prev, z_thresh, seed, seed_split
            )
            aucs=evaluate_auc(X,y,splits)
            if aucs["test"]>=auc_thr:
                # reconstruct alpha/beta
                rng=np.random.RandomState(seed)
                indep=independent_paths(paths)
                pool20=rng.choice(indep,20,replace=False).tolist()
                alpha={g:rng.normal(0,5) for g in X.columns}
                beta ={p:(rng.normal(0,5) if p in true10 else 0.0) for p in pool20}

                outdir=Path("simulation")/f"att{attempt}"/rel
                save_all(outdir,X,y,splits,alpha,beta,true10,false10,rel,seed,seed_split)
                print(f"🎉 Success attempt {attempt}, rel={rel}. Saved to {outdir}")
                return
    raise RuntimeError(f"No test-AUC ≥{auc_thr} in {max_iter} attempts")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--relationship",nargs="+",choices=["linear","nonlinear"],
                   default=["linear","nonlinear"])
    p.add_argument("--prev",type=float,default=0.5)
    p.add_argument("--out_z",type=float,default=5.0)
    p.add_argument("--auc_thr",type=float,default=0.75)
    p.add_argument("--max_iter",type=int,default=100)
    p.add_argument("--seed_base",type=int,default=10,
                   help="Base for seed (seed=seed_base+attempt)")
    p.add_argument("--seed_split",type=int,default=1000,
                   help="Random state for train_test_split")
    p.add_argument("--mut_fp",type=Path,
                   default=Path("./data/prostate/P1000_final_analysis_set_cross_important_only.csv"))
    p.add_argument("--gmt_fp",type=Path,
                   default=Path("./biological_knowledge/reactome/ReactomePathways.gmt"))
    args=p.parse_args()

    mut_df=load_mutation(args.mut_fp)
    paths=load_gmt(args.gmt_fp)
    run_until_success(
        args.relationship, mut_df, paths,
        args.prev, args.out_z, args.auc_thr, args.max_iter,
        args.seed_base, args.seed_split,
        args.mut_fp, args.gmt_fp
    )
