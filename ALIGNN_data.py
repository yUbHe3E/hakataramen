import os
import re
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from torch.utils.data import Dataset

# 清理字符串并转换浮点数
def clean_float(value):
    """ 处理形如 '13.6770(0)' 的字符串，将其转换为浮点数 """
    clean_value = re.sub(r'\(.*\)', '', value)
    return float(clean_value)

# 从CIF文件读取原子坐标和晶胞参数
def read_cif(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    a = b = c = alpha = beta = gamma = None
    atomic_species = []
    fractional_coords = []
    atom_site_started = False

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('_cell_length_a'):
            a = clean_float(line.split()[1])
        elif line.startswith('_cell_length_b'):
            b = clean_float(line.split()[1])
        elif line.startswith('_cell_length_c'):
            c = clean_float(line.split()[1])
        elif line.startswith('_cell_angle_alpha'):
            alpha = clean_float(line.split()[1])
        elif line.startswith('_cell_angle_beta'):
            beta = clean_float(line.split()[1])
        elif line.startswith('_cell_angle_gamma'):
            gamma = clean_float(line.split()[1])
        elif line.startswith('_atom_site_label'):
            atom_site_started = True
        elif atom_site_started and line:
            parts = line.split()
            if len(parts) >= 5:
                atomic_species.append(parts[1])
                fractional_coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

    if not all([a, b, c, alpha, beta, gamma]):
        raise ValueError("Missing cell parameters in CIF file.")

    return atomic_species, np.array(fractional_coords), a, b, c

# 定义范德华半径（单位: Å）
van_der_waals_radius = {
    'Al': 1.84,  # 铝
    'C': 1.70,   # 碳
    'N': 1.55,   # 氮
    'O': 1.52,   # 氧
    'O2-(H2O)': 1.52,  # 复杂形式，基于氧原子的范德华半径
    'P': 1.80,   # 磷
    'Si': 2.10   # 硅
}


def get_vdw_radius(atom):
    """ 根据原子符号返回范德华半径 """
    return van_der_waals_radius.get(atom, 1.7)  # 如果原子类型未定义，使用默认值 1.7

# 处理单个CIF文件并生成图数据
def process_cif_file(file_path, species_to_onehot):
    atomic_species, fractional_coords, a, b, c = read_cif(file_path)

    # 将分数坐标转换为笛卡尔坐标
    cartesian_coords = np.array([[a * x, b * y, c * z] for x, y, z in fractional_coords])
    node_features = np.array([species_to_onehot[species] for species in atomic_species])
    print(f"Fractional coordinates: {fractional_coords}")
    print(f"Cartesian coordinates: {cartesian_coords}")

    edge_index = []
    num_atoms = len(cartesian_coords)

    # 生成 edge_index，基于范德华半径
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(cartesian_coords[i] - cartesian_coords[j])

            # 获取两种原子之间的范德华半径
            radius_i = get_vdw_radius(atomic_species[i])
            radius_j = get_vdw_radius(atomic_species[j])
            bond_threshold = radius_i + radius_j  # 两个原子的范德华半径之和

            # 如果两原子之间的距离小于范德华半径的总和，就添加一条边
            if dist < bond_threshold:
                edge_index.append([i, j])

    edge_index = np.array(edge_index).T

    # 创建线图的边
    line_graph_edges = []
    for k in range(edge_index.shape[1]):
        for l in range(k + 1, edge_index.shape[1]):
            if len(set(edge_index[:, k]).intersection(set(edge_index[:, l]))) > 0:
                line_graph_edges.append([k, l])

    line_graph_edges = np.array(line_graph_edges).T

    # 转换为 PyTorch tensor 格式
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    line_graph_edges = torch.tensor(line_graph_edges, dtype=torch.long)

    # 生成 edge_index 后检查
    # max_index = max(edge_index.flatten().tolist())
    # num_atoms = len(cartesian_coords)
    # if max_index >= num_atoms:
    #     print(f"edge_index contains an out-of-bound index {max_index}, but only {num_atoms} atoms are available.",file_path)
    #     # raise ValueError(
    #     #     f"edge_index contains an out-of-bound index {max_index}, but only {num_atoms} atoms are available.")
    #
    # # 生成 line_graph_edge_index 后检查
    # max_line_graph_index = max(line_graph_edges.flatten().tolist())
    # if max_line_graph_index >= num_atoms:
    #     print(f"line_graph_edge_index contains an out-of-bound index {max_line_graph_index}, but only {num_atoms} atoms are available.",
    #           file_path)
    #     # raise ValueError(
    #     #     f"line_graph_edge_index contains an out-of-bound index {max_line_graph_index}, but only {num_atoms} atoms are available.")

    return Data(x=node_features, edge_index=edge_index, line_graph_edge_index=line_graph_edges)

# 动态构建原子类型集
def build_unique_species(directory):
    species_set = set()
    for filename in os.listdir(directory):
        if filename.endswith(".cif") or filename.endswith(".cif_"):
            file_path = os.path.join(directory, filename)
            atomic_species, _, _, _, _ = read_cif(file_path)
            species_set.update(atomic_species)
    print(sorted(list(species_set)))
    return sorted(list(species_set))



# 动态构建原子类型集
def build_unique_species(directory):
    """ 从所有CIF文件中构建统一的原子类型集 """
    species_set = set()

    for filename in os.listdir(directory):
        if filename.endswith(".cif") or filename.endswith(".cif_"):
            file_path = os.path.join(directory, filename)

            # 读取 CIF 文件以提取原子类型
            atomic_species, _, _, _, _ = read_cif(file_path)
            print(atomic_species)
            print(file_path)
            species_set.update(atomic_species)



    return sorted(list(species_set))


# 处理所有CIF文件
def process_all_cif_files(directory):
    """ 处理目录下的所有CIF文件，返回列表形式的PyTorch Geometric数据对象 """
    # 第一步：动态构建统一的原子类型集
    unique_species = build_unique_species(directory)
    print("Unique atomic species:", unique_species)

    # 为每种原子类型创建独热编码映射
    species_to_onehot = {species: np.eye(len(unique_species))[i] for i, species in enumerate(unique_species)}

    # 第二步：处理每个CIF文件并构建图数据
    data_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".cif") or filename.endswith(".cif_"):
            file_path = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            data = process_cif_file(file_path, species_to_onehot)
            if data:
                data_list.append(data)

    return data_list


# CIF文件所在目录
cif_directory = './cif_file/'

# 处理所有CIF文件
data_list = process_all_cif_files(cif_directory)

# 打印处理后的数据
for data in data_list:
    print(data)
