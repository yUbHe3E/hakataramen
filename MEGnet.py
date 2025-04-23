import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_mean

"""MEGNet implementation tailored for zeolite adsorption prediction.
- node_dim: dimension of node features (your one‑hot + any phys‑chem descriptors)
- edge_dim: dimension of edge attributes (Gaussian distance expansion length)
- state_dim: 2 (T & P) + n_adsorbate, or any embedding length you choose
"""

class MEGNetBlock(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, state_dim: int, hidden: int = 64):
        super().__init__()
        # e' = MLP_e([e, v_i, v_j, u])
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim + state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, edge_dim),
        )
        # v' = MLP_v([v, agg_e', u])
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, node_dim),
        )
        # u' = MLP_u([u, mean(v'), mean(e')])
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim + node_dim + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index  # message from row ➜ col
        # ── Edge update ─────────────────────────────────────────
        u_e = u[batch[row]]                 # 对齐到每条边所属图
        e_input = torch.cat([edge_attr, x[row], x[col], u_e], dim=1)
        e_out = self.edge_mlp(e_input)
        # ── Node update ─────────────────────────────────────────
        agg = scatter_mean(e_out, col, dim=0, dim_size=x.size(0))
        u_v = u[batch]
        v_input = torch.cat([x, agg, u_v], dim=1)
        v_out = self.node_mlp(v_input)
        # ── State update ────────────────────────────────────────
        v_graph = scatter_mean(v_out, batch, dim=0, dim_size=u.size(0))
        e_graph = scatter_mean(e_out, batch[row], dim=0, dim_size=u.size(0))
        u_input = torch.cat([u, v_graph, e_graph], dim=1)
        u_out  = self.state_mlp(u_input)
        return v_out, e_out, u_out


class MEGNet(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, state_dim: int,
                 hidden_dim: int = 64, n_blocks: int = 3, readout_hidden: int = 64):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, node_dim)
        self.edge_proj = nn.Linear(edge_dim, edge_dim)
        self.blocks = nn.ModuleList([
            MEGNetBlock(node_dim, edge_dim, state_dim, hidden_dim)
            for _ in range(n_blocks)
        ])
        self.readout = nn.Sequential(
            nn.Linear(node_dim + state_dim, readout_hidden),
            nn.ReLU(),
            nn.Linear(readout_hidden, 1),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr, u, batch = data.edge_attr, data.temp_pressure, data.batch
        # 预投影
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        # 堆叠 N 个 MEGNetBlock
        for blk in self.blocks:
            x, edge_attr, u = blk(x, edge_index, edge_attr, u, batch)
        # 图级池化
        x_pool = global_mean_pool(x, batch)
        out = self.readout(torch.cat([x_pool, u], dim=1))
        return out.view(-1)


"""=============== 使用方式（示例） ===============
from torch_geometric.loader import DataLoader
from your_dataset_file import AdsorptionDataset
from megnet_model import MEGNet

# 1) 数据集加载（沿用你的数据处理代码，需做两处小修改）
#   • 在生成 edge_index 时同时收集 distance 列表，使用 GaussianDistance 展开成 edge_attr
#   • 把已有的 graph_data.temp_pressure 赋给 graph_data.u
# -------------------------------------------------
# 2) 创建 dataloader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3) 构建模型
sample = dataset[0]
model = MEGNet(node_dim=sample.x.size(1),
               edge_dim=sample.edge_attr.size(1),
               state_dim=sample.u.size(0)).to(device)

# 4) 训练循环与 GINtrain.py 基本相同：
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion  = nn.MSELoss()
for epoch in range(EPOCHS):
    ...
"""
