import numpy as np
import torch
from torch_geometric.data import Data

# 原子种类，按照唯一原子种类表示
atomic_species = ['Li', 'Si', 'Al', 'O', 'O', 'O', 'O', 'H', 'H', 'O']

# 独热编码处理原子种类，每个原子类型只出现一次
unique_species = ['Li', 'Si', 'Al', 'O', 'H']
species_to_onehot = {species: np.eye(len(unique_species))[i] for i, species in enumerate(unique_species)}
node_features = np.array([species_to_onehot[species] for species in atomic_species])

# 从 CIF 内容中提取的分数坐标
fractional_coords = np.array([
    [0.1862, 0.6849, 0.252],  # Li
    [0.3544, 0.3757, 0.2492],  # Si
    [0.1593, 0.081, 0.25],  # Al
    [0.0061, 0.1584, 0.197],  # O1
    [0.2736, 0.2198, 0.1391],  # O2
    [0.1912, 0.0399, 0.5907],  # O3
    [0.1804, -0.1008, 0.0689],  # O4
    [0.5497, 0.1798, 0.8651],  # H1
    [0.4965, 0.0587, 0.5773],  # H2
    [0.4891, 0.0903, -0.2395],  # O5
])

# 晶胞参数 (a, b, c)
a, b, c = 10.313, 8.194, 4.993

# 将分数坐标转换为笛卡尔坐标
cartesian_coords = np.array([
    [a * x, b * y, c * z] for x, y, z in fractional_coords
])

# 根据距离阈值生成边索引（例如 3.0 Å）
edge_index = []
threshold = 3.0
num_atoms = len(cartesian_coords)

for i in range(num_atoms):
    for j in range(i + 1, num_atoms):
        dist = np.linalg.norm(cartesian_coords[i] - cartesian_coords[j])
        if dist < threshold:
            edge_index.append([i, j])

# 将数据转换为 PyTorch tensor 格式
node_features = torch.tensor(node_features, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 构建 PyTorch Geometric 数据对象
data = Data(x=node_features, edge_index=edge_index)

# 打印data对象
print(data)
