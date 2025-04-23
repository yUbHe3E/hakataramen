import os, time, json, math, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch_geometric.loader import DataLoader
from copy import deepcopy

from GIN import FullModel
from MEGnet import MEGNet
from GINdataset_1_1 import AdsorptionDataset as GINDataset
from GINdataset_megnet import AdsorptionDataset as MEGDataset

# ---------- 全局配置 ----------
choose_model      = "GIN"            # 'GIN' or 'MEGnet'
csv_all           = "newdata/modeldata/new_database_323.xlsx"
csv_test_ext      = "newdata/modeldata/testdata.xlsx"
cif_dir           = "./cif_file/"
cif_dict_path     = "./temp/graphs_dict_MEGnet.pt" if choose_model == "MEGNet" else "./temp/graphs_dict.pt"
adsorbate_map_path= "./temp/adsorbate_map.pt"
typical_ads       = ["CO2","CH4","N2"]  # 想另外保存预测曲线的气体
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 50
out_dir = "cv_results"; os.makedirs(out_dir, exist_ok=True)

# 归一化参数 (保持与你代码一致)
mean_tp = torch.tensor([1.2933904e+05, -6.1430967e-01]); std_tp = torch.tensor([4.47078413e+04, 1.98385120e+00])
mean_y  = torch.tensor(0.1904772226964999); std_y  = torch.tensor(1.317380664375389)
λT, λP, λY = 2.197387215938357, 0.09754163115529348, 0.0829811630928598

# ——— 工具：Dataset / Model / 单 epoch ———
def get_ds(model, mask):
    D = MEGDataset if model=="MEGnet" else GINDataset
    return D(csv_all, cif_dir, mean_tp, std_tp, mean_y, std_y,
             Temp_lambda=λT, pressure_lambda=λP, adsorption_lambda=λY,
             handle=True, test=False, cif_dict_path=cif_dict_path, adsorbate_map_path=adsorbate_map_path,
             idx_mask=mask)
def build(model, sample):
    if model=="GIN":
        return FullModel(12,8,2,8,15).to(device)
    return MEGNet(sample.x.size(1), sample.edge_attr.size(1),
                  sample.temp_pressure.size(1), hidden_dim=2, n_blocks=1).to(device)

loss_fn = nn.MSELoss()
def step(model, loader, opt=None, name="MEGnet"):
    tr = opt is not None; model.train() if tr else model.eval()
    pr, gt, Tlist, Plist = [], [], [], []; tot, lsum = 0, 0
    for d in loader:
        d = d.to(device)
        if name=="GIN":
            tp = d.temp_pressure.view(d.num_graphs, -1).to(device)
            out = model(d, tp); tp_cpu = tp.cpu()
        else:
            out = model(d); tp_cpu = d.temp_pressure.cpu()
        loss = loss_fn(out, d.y.view(-1))
        if tr: opt.zero_grad(); loss.backward(); opt.step()
        lsum += loss.item()*d.num_graphs; tot += d.num_graphs
        pr.append(out.detach().cpu()); gt.append(d.y.cpu())
        Tlist.append(tp_cpu[:,0]);     Plist.append(tp_cpu[:,1])
    return lsum/tot, torch.cat(pr), torch.cat(gt), torch.cat(Tlist), torch.cat(Plist)
# ——— 保存 .npz ———
def save_npz(path, p, t, T, P): np.savez(path, pred=p.numpy(), real=t.numpy(), T=T.numpy(), P=P.numpy())
# ——— 主循环 ———
def run(model_name, kA_list, ratios):
    outdir="cv_temp"; os.makedirs(outdir, exist_ok=True)
    raw = pd.read_excel(csv_all)
    all_cases_ext, all_cases_int = [], []  # 保存 (tag, rmse)
    # 1) Few‑shot‑A
    # for kA in kA_list:
    #     for ads in raw['adsorbate'].unique():
    #         rows_ads = raw.index[raw['adsorbate']==ads]
    #         if len(rows_ads)<=kA: continue
    #         keep = np.random.choice(rows_ads, kA, replace=False)
    #         m_train = raw.index.isin(keep) | (raw['adsorbate']!=ads)
    #         m_test  = raw.index.isin(rows_ads)&~raw.index.isin(keep)
    #         tr_ds, te_ds = get_ds(model_name,m_train), get_ds(model_name,m_test)
    #         model = build(model_name,tr_ds[0]); opt=optim.Adam(model.parameters(),lr=1e-3)
    #         best,best_state=float("inf"),None
    #         for _ in range(epoch):
    #             step(model, DataLoader(tr_ds,32,shuffle=True), opt, model_name)
    #             val,_p,_t,_,_ = step(model, DataLoader(te_ds,64), None, model_name)
    #             if val<best: best,best_state=val,deepcopy(model.state_dict())
    #         model.load_state_dict(best_state)
    #         _,p,t,Tvec,Pvec = step(model, DataLoader(te_ds,64),None,model_name)
    #         rmse = torch.sqrt(((p-t)**2).mean()).item()
    #         tag  = f"fewshotA_{ads}_K{kA}"
    #         save_npz(f"{outdir}/{tag}.npz", p,t,Tvec,Pvec)
    #         print(f"[Few‑shot‑A] {ads:<4s} K_A={kA} RMSE={rmse:.3f}")
    # 2) LOTO‑T with ratio anchor on extremes
    for ratio in ratios:
        for (zeo,ads), sub in raw.groupby(["zeolite_type","adsorbate"]):
            temps = sorted(sub['temperature'].unique())
            if len(temps)<2: continue
            Tmin,Tmax = temps[0], temps[-1]
            for left_T in temps:
                rows_left = sub.index[sub['temperature']==left_T]
                # 是否极端温度
                is_extreme = left_T in (Tmin,Tmax)
                n_anchor = max(1,int(len(rows_left)*ratio)) if is_extreme and ratio>0 else 0
                keep = np.random.choice(rows_left, n_anchor, replace=False) if n_anchor>0 else []
                m_train = raw.index.isin(keep) | ~raw.index.isin(rows_left)
                m_test  = raw.index.isin(rows_left)&~raw.index.isin(keep)
                tr_ds, te_ds = get_ds(model_name,m_train), get_ds(model_name,m_test)
                model = build(model_name,tr_ds[0]); opt=optim.Adam(model.parameters(),lr=1e-3)
                best,bstate=float("inf"),None
                for _ in range(epoch):
                    step(model, DataLoader(tr_ds,512,shuffle=True),opt,model_name)
                    v,_p,_t,_,_ = step(model,DataLoader(te_ds,512),None,model_name)
                    if v<best: best,bstate=v,deepcopy(model.state_dict())
                model.load_state_dict(bstate)
                _,p,t,Tv,Pv = step(model,DataLoader(te_ds,512),None,model_name)
                rmse = torch.sqrt(((p-t)**2).mean()).item()
                kind = "EX" if is_extreme else "IN"
                tag  = f"lotoT_{zeo}_{ads}_{left_T}K_{kind}_r{ratio}"
                save_npz(f"{outdir}/{tag}.npz", p,t,Tv,Pv)
                print(f"[{kind}] {zeo}/{ads} leave {left_T}K r={ratio} RMSE={rmse:.3f}")
                (all_cases_ext if is_extreme else all_cases_int).append((tag,rmse))
    # 3) 选典型案例 (最佳/中位/最差)
    def pick_cases(cases, label):
        if not cases: return
        cases = sorted(cases, key=lambda x:x[1])
        picks = [cases[0], cases[len(cases)//2], cases[-1]]
        for tag,_ in picks:
            os.rename(f"{outdir}/{tag}.npz", f"{outdir}/TYP_{label}_{tag}.npz")
    pick_cases(all_cases_ext,"EX"); pick_cases(all_cases_int,"IN")

# ——— CLI ———
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=["GIN","MEGnet"],default="MEGnet")
    ap.add_argument("--kA",type=int,nargs="+",default=[0,2],help="few‑shot lines for adsorbate")
    ap.add_argument("--ratio",type=float,nargs="+",default=[0.0,0.1],help="anchor ratio for extreme temperatures")
    args=ap.parse_args()
    run(args.model,args.kA,args.ratio)
    # run(args.model,args.ratio)
