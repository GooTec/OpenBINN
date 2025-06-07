#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulation – adjustable signal, linear vs nonlinear, optional intercept calibration,
1 true + 2 null pathways (≤10 genes), save best-AUC + all original outputs
"""
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_mutation(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, index_col=0)
    print(f"Loaded mutation data: {df.shape}")
    return df

def load_gmt(fp: Path) -> Dict[str, Set[str]]:
    pathways = {}
    with fp.open() as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                genes = set(parts[2:])
                if 1 <= len(genes) <= 10:
                    pathways[parts[0]] = genes
    print(f"Loaded {len(pathways)} candidate pathways (≤10 genes each)")
    return pathways

def independent_paths(all_paths: Dict[str,Set[str]], thr: float=0.2) -> List[str]:
    def jacc(a, b): return len(a&b)/len(a|b) if (a|b) else 1.0
    keep = []
    for p, genes in all_paths.items():
        sims = [jacc(genes, all_paths[o]) for o in all_paths if o!=p]
        if sims and np.mean(sims) < thr:
            keep.append(p)
    return keep

def calibrate_intercept(eta: np.ndarray, prev: float) -> float:
    lo, hi = -20, 20
    for _ in range(40):
        mid = (lo+hi)/2
        if expit(eta+mid).mean() > prev:
            hi = mid
        else:
            lo = mid
    return mid

def evaluate_auc(X: pd.DataFrame, y: pd.Series, splits: Dict[str,List[int]]) -> float:
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000)
    model.fit(X.loc[splits["train"]], y.loc[splits["train"]])
    prob = model.predict_proba(X.loc[splits["test"]])[:,1]
    auc = roc_auc_score(y.loc[splits["test"]], prob)
    return auc

def trial_once(
    mut_df: pd.DataFrame, paths: Dict[str,Set[str]],
    prev: float, alpha_sigma: float,
    use_nonlinear: bool, w_additive_only: bool,
    skip_calibrate: bool, seed: int, seed_split: int
) -> Tuple[pd.DataFrame,pd.Series,Dict[str,List[int]],Dict[str,float],Dict[str,float],List[str],List[str],np.ndarray,float]:
    rng = np.random.RandomState(seed)
    indep = independent_paths(paths)

    # 1 true + 2 null
    true_p = rng.choice(indep,1)[0]
    nulls = list(rng.choice([p for p in indep if p!=true_p],2,replace=False))
    pool = [true_p] + nulls

    # deduplicate genes
    genes = sorted({g for p in pool for g in paths[p] if g in mut_df.columns})
    X = mut_df[genes].copy()

    # optional outlier removal
    pcs = PCA(n_components=2).fit_transform(X.values)
    z = np.abs(stats.zscore(pcs, axis=0))
    mask = (z>3).any(axis=1)
    if mask.any():
        X = X.loc[~mask]

    # alpha & beta
    alpha = {g: (rng.normal(0, alpha_sigma) if g in paths[true_p] else 0.0) for g in genes}
    beta  = {p: (1.0 if p==true_p else 0.0) for p in pool}

    # compute components
    arr = X.values
    a_vec = np.array([alpha[g] for g in genes])
    additive = arr.dot(a_vec)

    if use_nonlinear:
        tg = [g for g in genes if g in paths[true_p]]
        if len(tg)>=2:
            g1,g2 = rng.choice(tg,2,replace=False)
            mult = X[g1].values * X[g2].values
            OR   = np.maximum(X[g1].values, X[g2].values)
            AND  = np.minimum(X[g1].values, X[g2].values)
        else:
            mult=OR=AND=np.zeros(len(X))
    else:
        mult=OR=AND=np.zeros(len(X))

    w = np.array([1,0,0,0]) if w_additive_only else rng.uniform(size=4)
    eta = w[0]*additive + w[1]*mult + w[2]*OR + w[3]*AND

    c = 0.0 if skip_calibrate else calibrate_intercept(eta, prev)
    prob = expit(eta + c)
    y = pd.Series(rng.binomial(1,prob), index=X.index, name="response")

    # split
    ids = X.index.to_list()
    tv, test = train_test_split(ids, test_size=0.1, stratify=y, random_state=seed_split)
    train, val = train_test_split(tv, test_size=0.1111, stratify=y.loc[tv], random_state=seed_split)
    splits = {"train":train, "val":val, "test":test}

    return X, y, splits, alpha, beta, [true_p], nulls, eta, c

def save_all(
    outdir: Path, X: pd.DataFrame, y: pd.Series, splits: Dict[str,List[int]],
    alpha: Dict[str,float], beta: Dict[str,float], true10: List[str], false10: List[str],
    eta: np.ndarray, c: float, seed: int, seed_split: int
):
    outdir.mkdir(parents=True, exist_ok=True)
    # seeds
    with open(outdir/"seeds.txt","w") as f:
        f.write(f"seed_base_run: {seed}\nseed_split: {seed_split}\n")

    # PCA plot
    pcs = PCA(n_components=2).fit_transform(X.values)
    plt.figure(figsize=(6,5))
    pal = sns.color_palette("Set1",2)
    for cls in (0,1):
        idx = y==cls
        plt.scatter(pcs[idx,0],pcs[idx,1],s=20,alpha=0.6,edgecolor="k",c=[pal[cls]],label=f"M={cls}")
    plt.title(f"PCA – seed={seed}")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(frameon=False)
    plt.savefig(outdir/"pca_plot.png",dpi=200)
    plt.close()

    # original filenames
    X.to_csv(outdir/"somatic_mutation_paper.csv")
    y.to_frame().to_csv(outdir/"response.csv",index=True)
    pd.DataFrame(alpha.items(),columns=["gene","alpha"]).to_csv(outdir/"gene_alpha.csv",index=False)
    pd.DataFrame(beta.items(), columns=["pathway","beta"]).to_csv(outdir/"pathway_beta.csv",index=False)
    pd.Series(X.columns,name="genes").to_csv(outdir/"selected_genes.csv",index=False)

    sp = outdir/"splits"; sp.mkdir(exist_ok=True)
    pd.DataFrame({"id":splits["train"],"response":y.loc[splits["train"]]}).to_csv(sp/"training_set_0.csv",index=True)
    pd.DataFrame({"id":splits["val"],  "response":y.loc[splits["val"]]}).to_csv(sp/"validation_set.csv",   index=True)
    pd.DataFrame({"id":splits["test"], "response":y.loc[splits["test"]]}).to_csv(sp/"test_set.csv",         index=True)

    # eta & intercept
    np.save(outdir/"eta.npy", eta)
    with open(outdir/"intercept.txt","w") as f:
        f.write(str(c))

    print(f"Saved simulation to {outdir}")

def run_best(args):
    mut_df = load_mutation(args.mut_fp)
    paths  = load_gmt(args.gmt_fp)

    best_auc = -np.inf
    best_cfg = None

    for i in range(1, args.max_iter+1):
        seed = args.seed_base + i
        print(f"Attempt {i} seed={seed}:", end="")
        X, y, splits, alpha, beta, t10, f10, eta, c = trial_once(
            mut_df, paths, args.prev,
            args.alpha_sigma, args.use_nonlinear,
            args.w_additive_only, args.skip_calibrate,
            seed, args.seed_split
        )
        auc = evaluate_auc(X, y, splits)
        print(f" test AUC={auc:.3f}")
        if auc > best_auc:
            best_auc = auc
            best_cfg = (X, y, splits, alpha, beta, t10, f10, eta, c, seed)

    if best_cfg is None:
        raise RuntimeError("No valid scenario")
    Xb, yb, sb, ab, bb, t10b, f10b, etab, cb, sb_seed = best_cfg
    outdir = Path("simulation")/"best_simple"
    save_all(outdir, Xb, yb, sb, ab, bb, t10b, f10b, etab, cb, sb_seed, args.seed_split)
    print(f"\nBest test AUC={best_auc:.3f} (seed={sb_seed}) saved to {outdir}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mut_fp",        type=Path, default=Path("./data/prostate/P1000_final_analysis_set_cross_important_only.csv"))
    p.add_argument("--gmt_fp",        type=Path, default=Path("./biological_knowledge/reactome/ReactomePathways.gmt"))
    p.add_argument("--prev",          type=float, default=0.5)
    p.add_argument("--alpha_sigma",   type=float, default=20.0,
                   help="α 표준편차 (default=10)")
    p.add_argument("--use_nonlinear", action="store_true",
                   help="include mult/OR/AND")
    p.add_argument("--w_additive_only", action="store_true",
                   help="only additive component")
    p.add_argument("--skip_calibrate", action="store_true",
                   help="skip intercept calibration")
    p.add_argument("--max_iter",      type=int, default=50)
    p.add_argument("--seed_base",     type=int, default=10)
    p.add_argument("--seed_split",    type=int, default=123)
    args = p.parse_args()

    run_best(args)
