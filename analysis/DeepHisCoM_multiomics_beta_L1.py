#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_beta_from_permutations.py
────────────────────────────────
β 추정:
  • original  (param0.txt)           → λ 튜닝
  • 100× permutation (param1..100)  → 튜닝된 λ 재사용
결과: results/{gene_permutation,label_permutation}/exp/<sim>/param<ID>.txt
"""

import warnings, argparse, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# ───────── 설정 ─────────
SEED = 100
N_VAR = 100       # permutation 1..100
LR    = 1e-3
PATIENCE, MAX_EPOCHS = 20, 10_000
L1_GRID = [0, 1e-6, 1e-5, 1e-4, 1e-3]   # λ 후보
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_ROOT = Path("./data/b0_g0.0")
RES_ROOT  = Path("./results/b0_g0.0")
GMT_FILE  = Path("../biological_knowledge/simulation/SimulationPathways_DeepHisCoM_L1.gmt")

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# λ 캐시: {(sim, variant): best_lambda}
BEST_L1 = {}

# ───────── 1. GMT ─────────
def load_gmt(fp: Path):
    ids, dic = [], {}
    for ln in open(fp):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) >= 3:
            ids.append(parts[0]); dic[parts[0]] = parts[2:]
    return ids, dic
PATHS, PATH2GENE = load_gmt(GMT_FILE)

# ───────── 2. 데이터 I/O ─────────
def feature_df(d: Path):
    som = pd.read_csv(d/"somatic_mutation_paper.csv", index_col=0)
    cnv = pd.read_csv(d/"P1000_data_CNA_paper.csv", index_col=0)
    delm = (cnv == -2).astype(int); ampl = (cnv == 2).astype(int)
    genes = sorted(set(som.columns).union(cnv.columns))
    X = np.stack([som[genes].fillna(0).values.astype(np.float32),
                  delm[genes].fillna(0).values.astype(np.float32),
                  ampl[genes].fillna(0).values.astype(np.float32)], axis=2)
    y = pd.read_csv(d/"response.csv", index_col=0)["response"].values.astype(np.float32)
    return X, y, genes, som.index.tolist()

def split_ids(d: Path):
    s = d/"splits"
    tr = pd.read_csv(s/"training_set_0.csv")["id"].tolist()
    va = pd.read_csv(s/"validation_set.csv")["id"].tolist()
    te = pd.read_csv(s/"test_set.csv")["id"].tolist()
    return tr, va+te

def idx_builder(genes):
    g2i = {g:i for i,g in enumerate(genes)}
    return {p:[g2i[g] for g in PATH2GENE.get(p,[]) if g in g2i] for p in PATHS}

# ───────── 3. Dataset ─────────
class MultiOmicsDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.from_numpy(X); self.y=torch.from_numpy(y).float().unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

def loader(X,y,shuffle):
    return DataLoader(MultiOmicsDataset(X,y),batch_size=len(y),shuffle=shuffle)

# ───────── 4. 모델 ─────────
class GeneEnc(nn.Module):
    def __init__(self): super().__init__(); self.lin=nn.Linear(3,1,bias=False)
    def forward(self,x): B,G,_=x.size(); return self.lin(x.view(B*G,3)).view(B,G)

class HisCoM(nn.Module):
    def __init__(self,n_genes,pid2idx):
        super().__init__(); self.pid2idx=pid2idx
        self.gene=GeneEnc()
        self.path=nn.ModuleDict({p:nn.Linear(len(idxs),1,bias=False)
                                 for p,idxs in pid2idx.items()})
        P=len(pid2idx); self.bn=nn.BatchNorm1d(P); self.fc=nn.Linear(P,1)
    def forward(self,x):
        B=x.size(0); g=self.gene(x)                   # [B,G]
        ps=[self.path[p](g[:,idxs]) if idxs else torch.zeros(B,1,device=x.device)
            for p,idxs in self.pid2idx.items()]
        Pmat=torch.cat(ps,1); return torch.sigmoid(self.fc(self.bn(Pmat)))

# ───────── 5. 학습 ─────────
def train_beta(Xd,pid2idx,l1_lambda):
    tr=loader(Xd["train_X"],Xd["train_y"],True)
    va=loader(Xd["val_X"],  Xd["val_y"],  False)
    net=HisCoM(Xd["train_X"].shape[1],pid2idx).to(DEVICE)
    opt=optim.Adam(net.parameters(),lr=LR); bce=nn.BCELoss()
    best_auc,best_beta,pat= -1,None,0
    for epoch in range(1,MAX_EPOCHS+1):
        net.train()
        for xb,yb in tr:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); out=net(xb)
            loss=bce(out,yb)+l1_lambda*sum(p.abs().sum() for p in net.parameters())
            loss.backward(); opt.step()
        
        # validation
        net.eval()
        with torch.no_grad():
            xb, yb = next(iter(va))
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            probs  = net(xb).detach().cpu().squeeze().numpy()   # ← detach
            labels = yb.detach().cpu().squeeze().numpy()
        auc = roc_auc_score(labels, probs)
        if auc>best_auc: 
            best_auc = auc 
            best_beta = net.fc.weight.detach().cpu().numpy().ravel()  
            pat = 0
        else: pat+=1
        if pat>PATIENCE: break
    return best_beta,best_auc

# λ 튜닝 (param0용)
def tune_hparam(Xd,pid2idx):
    best_auc,best_beta,best_l1 = -1,None,None
    for lam in L1_GRID:
        beta,auc = train_beta(Xd,pid2idx,lam)
        if auc>best_auc: best_auc,best_beta,best_l1 = auc,beta,lam
    return best_beta,best_auc,best_l1

# ───────── 6. save_beta ─────────
def save_beta(sim:int, variant:str, vid:int):
    out_dir = RES_ROOT/variant/"exp"/f"{sim}"; out_dir.mkdir(parents=True,exist_ok=True)
    out_fp  = out_dir/f"param{vid}.txt"
    if out_fp.exists(): return

    dpath = DATA_ROOT/f"{sim}" if vid==0 else DATA_ROOT/f"{sim}"/variant.replace("_","-")/f"{vid}"
    if not dpath.exists(): print(f"  [skip] {dpath}"); return

    X_all,y_all,genes,samples = feature_df(dpath)
    tr_ids,va_ids = split_ids(dpath)
    idx_map = {s:i for i,s in enumerate(samples)}
    try:
        tr_idx=[idx_map[s] for s in tr_ids]; va_idx=[idx_map[s] for s in va_ids]
    except KeyError as e:
        print("  [skip] sample mismatch:",e); return
    Xd=dict(train_X=X_all[tr_idx],train_y=y_all[tr_idx],
            val_X=X_all[va_idx],  val_y=y_all[va_idx])
    pid2idx = idx_builder(genes)

    key=(sim,variant)
    # --- param0: 튜닝
    if vid==0:
        beta,auc,best_l1 = tune_hparam(Xd,pid2idx)
        BEST_L1[key]=best_l1
        np.savetxt(out_fp,beta,fmt="%.6g")
        print(f"  → {out_fp.relative_to(RES_ROOT)} | val AUC={auc:.4f} | λ={best_l1:g}")
    # --- permutation: 저장된 λ 재사용
    else:
        lam = BEST_L1.get(key, L1_GRID[0])   # fallback
        beta,auc = train_beta(Xd,pid2idx,lam)
        np.savetxt(out_fp,beta,fmt="%.6g")
        print(f"  → {out_fp.relative_to(RES_ROOT)} | val AUC={auc:.4f} | λ={lam:g}")

# ───────── 7. main ─────────
def main():
    warnings.filterwarnings("ignore")
    ap=argparse.ArgumentParser()
    ap.add_argument("--start_sim",type=int,default=1)
    ap.add_argument("--end_sim",  type=int,default=100)
    args=ap.parse_args()

    for s in range(args.start_sim,args.end_sim+1):
        print(f"\n■■ Simulation {s:3d}")
        for v in ["label_permutation"]:
            save_beta(s,v,0)                     # param0 (튜닝)
            for b in range(1,N_VAR+1):
                save_beta(s,v,b)                 # permutation (λ 재사용)
    print("\n✓ β 텍스트 저장 완료.")

if __name__=="__main__":
    main()
