#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_deephiscom_pvalues.py   (v2)
────────────────────────────
β 텍스트(param*.txt) → null-distribution & p-value
  · gene_permutation   +  label_permutation   모두 처리
"""

from pathlib import Path
import argparse, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

# ───── 설정 ─────
N_PERM      = 100
RESULT_ROOT = Path("./results/b0_g0.0")
GMT_FILE    = "../biological_knowledge/simulation/SimulationPathways_DeepHisCoM_L1.gmt"
VARIANTS    = ["label_permutation"] 
MAX_PLOT    = 10
sns.set(style="whitegrid")
#────────────────

def load_gmt(fp):
    ids=[]
    with open(fp) as f:
        for ln in f:
            p=ln.rstrip("\n").split("\t")
            if len(p)>=3: ids.append(p[1])
    return ids
PATHWAYS = load_gmt(GMT_FILE)

def process(sim, variant):
    exp_dir = RESULT_ROOT/variant/"exp"/f"{sim}"
    if not (exp_dir/"param0.txt").exists():
        print(f"[skip] {variant} sim {sim}: param0.txt 없음"); return

    obs = np.abs(np.loadtxt(exp_dir/"param0.txt"))
    null=[]
    for k in range(1, N_PERM+1):
        f=exp_dir/f"param{k}.txt"
        if f.exists(): null.append(np.loadtxt(f))
    if not null:
        print(f"[skip] {variant} sim {sim}: permutation β 없음"); return
    null=np.abs(np.vstack(null))
    P=min(len(PATHWAYS), obs.size, null.shape[1])
    obs,null=obs[:P],null[:,:P]
    pvals=(null>=obs).mean(0)

    # CSV
    out_csv=RESULT_ROOT/variant/f"sim_{sim}_pvalues_DeepHisCoM.csv"
    pd.DataFrame({'pathway':PATHWAYS[:P],'p_value':pvals}).to_csv(out_csv,index=False)

    # 그림
    top=np.argsort(pvals)[:MAX_PLOT]
    fig,axs=plt.subplots(len(top),1,figsize=(8,2.5*len(top)),sharex=True)
    xmin,xmax= null.min(),null.max(); xmin=min(xmin,obs.min()); xmax=max(xmax,obs.max())
    rng=.05*(xmax-xmin); xmin-=rng; xmax+=rng
    for ax,i in zip(axs,top):
        sns.histplot(null[:,i],bins=40,stat='density',kde=True,ax=ax)
        ax.axvline(obs[i],color='red',ls='--',label=f'|β₀|={obs[i]:.2g}')
        ax.set_xlim(xmin,xmax); ax.set_title(f"{PATHWAYS[i]} (p={pvals[i]:.3g})")
        ax.legend(frameon=False); ax.set_ylabel('Density')
    axs[-1].set_xlabel('|β|'); plt.tight_layout()
    out_png=RESULT_ROOT/variant/f"sim_{sim}_distributions_DeepHisCoM.png"
    plt.savefig(out_png,dpi=300); plt.close()
    print(f"{variant:<18} sim {sim:3d} │ 저장 완료")

def main():
    warnings.filterwarnings("ignore")
    ap=argparse.ArgumentParser()
    ap.add_argument("--start_sim",type=int,default=1)
    ap.add_argument("--end_sim",type=int,default=100)
    args=ap.parse_args()
    for sim in range(args.start_sim, args.end_sim+1):
        for var in VARIANTS:
            process(sim,var)
    print("\n✓ 두 permutation 모두 p-value/그림 계산 완료.")

if __name__=="__main__":
    main()
