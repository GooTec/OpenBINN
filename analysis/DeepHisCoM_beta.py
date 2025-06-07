#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_beta_from_permutations.py
────────────────────────────────
β 추정:
  • original  →  param0.txt
  • 100 × gene-perm / label-perm → param1..100.txt

β 텍스트는  results/{gene_permutation,label_permutation}/exp/ 에 저장
"""

import warnings, argparse, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# ────────── 하이퍼 파라미터 ──────────
SEED, N_VAR, LR, HIDDEN = 100, 100, 1e-3, 32
PATIENCE, MAX_EPOCHS = 5, 10_000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("./data/b4_g4")
RES_ROOT  = Path("./results/b4_g4/")
GMT_FILE  = "../biological_knowledge/simulation/SimulationPathways.gmt"
SUF = ("_som", "_del", "_amp")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ────────── GMT 로드 ──────────
def load_gmt(fp):
    ids, d = [], {}
    with open(fp) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                ids.append(p[0]); d[p[0]] = p[2:]
    return ids, d
PATHS, PATH2GENE = load_gmt(GMT_FILE)

# ────────── Dataset ──────────
class ArrDS(Dataset):
    def __init__(self, df):
        a = df.values.astype(np.float32)
        self.x, self.y = a[:,:-1], a[:,-1].reshape(-1,1)
    def __len__(self): return len(self.y)
    def __getitem__(s, i): return s.x[i], s.y[i]
def loader(df): return DataLoader(ArrDS(df), batch_size=len(df), shuffle=True)

# ────────── 모델 ──────────
def block(d): return nn.Sequential(nn.Linear(d,1,bias=False),
                                   nn.LeakyReLU(0.2))
class HisCoM(nn.Module):
    def __init__(s, nvar, idx):
        super().__init__(); s.idx = idx
        s.mods = nn.ModuleList([block(n) for n in nvar])
        s.bn  = nn.BatchNorm1d(len(nvar)); s.fc = nn.Linear(len(nvar),1)
    def forward(s,x):
        reps = [s.mods[i](x[:, ix]) for i,ix in enumerate(s.idx)]
        return torch.sigmoid(s.fc(s.bn(torch.cat(reps,1))))

# ────────── 유틸 ──────────
def feature_df(d:Path):
    som = pd.read_csv(d/"somatic_mutation_paper.csv", index_col=0)
    cnv = pd.read_csv(d/"P1000_data_CNA_paper.csv", index_col=0)
    delm = (cnv==-2).astype(int); ampm=(cnv==2).astype(int)
    genes = som.columns.intersection(cnv.columns)
    feat = pd.concat([som[genes].add_suffix("_som"),
                    delm[genes].add_suffix("_del"),
                    ampm[genes].add_suffix("_amp")], axis=1)
    y = pd.read_csv(d/"response.csv", index_col=0)['response']
    feat['phenotype'] = y.values; return feat
def split_ids(d:Path):
    s=d/"splits"
    tr = pd.read_csv(s/"training_set_0.csv")['id'].tolist()
    va = pd.read_csv(s/"validation_set.csv")['id'].tolist()
    te = pd.read_csv(s/"test_set.csv")['id'].tolist()
    return tr,va+te
def idx_builder(cols):
    c2i = {c:i for i,c in enumerate(cols)}
    idx,nv=[],[]
    for pid in PATHS:
        arr=[c2i[f"{g}{s}"] for g in PATH2GENE[pid] for s in SUF
             if f"{g}{s}" in c2i]
        idx.append(torch.tensor(arr,dtype=torch.long)); nv.append(len(arr))
    return idx,nv

def train_beta(tr_df,va_df,idx,nv):
    tr,va=loader(tr_df),loader(va_df)
    net=HisCoM(nv,idx).to(DEVICE)
    opt=optim.Adam(net.parameters(), lr=LR); crit=nn.BCELoss()
    best,auc,pat=None,-1,0
    for _ in range(MAX_EPOCHS):
        net.train()
        for xb,yb in tr:
            xb,yb=xb.to(DEVICE), yb.squeeze().to(DEVICE)
            opt.zero_grad(); loss=crit(net(xb).squeeze(), yb); loss.backward(); opt.step()
        net.eval(); scr,lab=[],[]
        with torch.no_grad():
            for xb,yb in va:
                xb,yb=xb.to(DEVICE), yb.squeeze().to(DEVICE)
                scr+=net(xb).squeeze().cpu().tolist()
                lab+=yb.cpu().tolist()
        a=roc_auc_score(lab,scr)
        if a>auc: auc,pat,best=a,0, net.fc.weight.detach().cpu().numpy().ravel()
        else: pat+=1
        if pat>PATIENCE: break
    return best

# ────────── β 저장 ──────────
def save_beta(sim, variant, vid):
    """
    sim      : 시뮬레이션 번호 (1…)
    variant  : 'gene_permutation' | 'label_permutation'
    vid      : 0 = original, 1–100 = permutation ID
    """
    # ① 저장 디렉터리 →  results/<variant>/exp/<sim_number>/
    out_dir = RES_ROOT / variant / "exp" / f"{sim}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ② 파일 이름       →  param<k>.txt
    out_fp = out_dir / f"param{vid}.txt"
    if out_fp.exists():
        return  # 이미 계산되어 있음

    # -------- 데이터 폴더 결정 --------
    dpath = (
        DATA_ROOT / f"{sim}"
        if vid == 0 else
        DATA_ROOT / f"{sim}" / variant.replace("_", "-") / f"{vid}"
    )
    if not dpath.exists():
        print(f"   [skip] {dpath} (폴더 없음)")
        return

    # -------- β 학습 및 저장 --------
    df          = feature_df(dpath)
    tr_ids, va_ids = split_ids(dpath)
    idx, nvar   = idx_builder(df.columns[:-1])
    beta_vec    = train_beta(df.loc[tr_ids], df.loc[va_ids], idx, nvar)

    np.savetxt(out_fp, beta_vec, fmt="%.6g")
    print(f"   → {out_fp.relative_to(RES_ROOT)} 저장 완료")

# ────────── main ──────────
def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_sim", type=int, default=1)
    ap.add_argument("--end_sim", type=int, default=100)
    args=ap.parse_args()

    for s in range(args.start_sim, args.end_sim+1):
        print(f"\n■■ Simulation {s:3d}")
        # original
        for v in ("gene_permutation","label_permutation"):
            save_beta(s, v, 0)
        # permutations
        for b in range(1, N_VAR+1):
            save_beta(s,"gene_permutation",b)
            save_beta(s,"label_permutation",b)

    print("\n✓ β 텍스트 저장 완료 (exp 폴더).")

if __name__=="__main__":
    main()
