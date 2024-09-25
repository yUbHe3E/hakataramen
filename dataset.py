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
    # 创建线图的边
    line_graph_edges = []
    num_edges = edge_index.shape[1]  # 原图中的边数量

    for k in range(num_edges):
        for l in range(k + 1, num_edges):
            if len(set(edge_index[:, k]).intersection(set(edge_index[:, l]))) > 0:
                line_graph_edges.append([k, l])

    line_graph_edges = np.array(line_graph_edges).T

    # 转换为 PyTorch tensor 格式
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    line_graph_edges = torch.tensor(line_graph_edges, dtype=torch.long)

    # 生成 edge_index 后检查
    max_index = max(edge_index.flatten().tolist())
    num_atoms = len(cartesian_coords)
    if max_index >= num_atoms:
        print(f"edge_index contains an out-of-bound index {max_index}, but only {num_atoms} atoms are available.",file_path)
        raise ValueError(
            f"edge_index contains an out-of-bound index {max_index}, but only {num_atoms} atoms are available.")

    # 生成 line_graph_edge_index 后检查
    max_line_graph_index = max(line_graph_edges.flatten().tolist())
    if max_line_graph_index >= num_edges:
        print(f"line_graph_edge_index contains an out-of-bound index {max_line_graph_index}, but only {num_atoms} atoms are available.",
              file_path)
        raise ValueError(
            f"line_graph_edge_index contains an out-of-bound index {max_line_graph_index}, but only {num_atoms} atoms are available.")

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

# 自定义 PyTorch 数据集类
class AdsorptionDataset(Dataset):
    def __init__(self, csv_file, cif_directory, mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption):
        self.data_frame = pd.read_excel(csv_file)  # 读取 Excel 文件
        self.cif_directory = cif_directory  # CIF 文件的目录
        self.mean_temp_pressure = mean_temp_pressure
        self.std_temp_pressure = std_temp_pressure
        self.mean_adsorption = mean_adsorption
        self.std_adsorption = std_adsorption

        # 构建统一的原子类型集
        self.unique_species = build_unique_species(cif_directory)
        self.species_to_onehot = {species: np.eye(len(self.unique_species))[i] for i, species in enumerate(self.unique_species)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        temp = row['Temp']
        pressure = row['Pressure(Bar)']
        adsorption = row['total_adsorption(mmol/g)']
        zeolite_type = row['zeolite_type']
        type = row['Type']



        # 匹配CIF文件
        cif_file = None
        for filename in os.listdir(self.cif_directory):
            if filename.startswith(zeolite_type) and filename.endswith(".cif_"):
                cif_file = os.path.join(self.cif_directory, filename)
                break

        if cif_file is None:
            raise FileNotFoundError(f"No CIF file found for zeolite type: {zeolite_type}")

        # 获取图数据
        graph_data = process_cif_file(cif_file, self.species_to_onehot)

        # 归一化温度、压力和type
        temp_pressure = torch.tensor([temp, pressure, type], dtype=torch.float)
        temp_pressure = (temp_pressure - self.mean_temp_pressure) / self.std_temp_pressure

        # 归一化 adsorption
        normalized_adsorption = (adsorption - self.mean_adsorption) / self.std_adsorption

        # 将归一化后的温度、压力和吸附量添加到 Data 对象中
        graph_data.temp_pressure = temp_pressure
        graph_data.y = torch.tensor(normalized_adsorption, dtype=torch.float)  # 吸附量作为标签

        return graph_data

def calculate_normalization_params(data_frame):
    temps = data_frame['Temp'].values
    pressures = data_frame['Pressure(Bar)'].values
    types = data_frame['Type'].values
    adsorptions = data_frame['total_adsorption(mmol/g)'].values

    # 计算温度、压力、type 和 adsorption 的均值和标准差
    mean_temp_pressure = np.mean([temps, pressures, types], axis=1)
    std_temp_pressure = np.std([temps, pressures, types], axis=1)
    mean_adsorption = np.mean(adsorptions)
    std_adsorption = np.std(adsorptions)

    return mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption
# 示例用法
if __name__ == '__main__':
    csv_file = 'database.xlsx'  # 包含温度、压力、吸附量和沸石种类的 Excel 文件
    cif_directory = './cif_file/'  # CIF 文件的目录
    mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption = calculate_normalization_params(
        pd.read_excel(csv_file))
    print(mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption)
    # 创建数据集实例
    adsorption_dataset = AdsorptionDataset(csv_file=csv_file, cif_directory=cif_directory,
                            mean_temp_pressure=mean_temp_pressure, std_temp_pressure=std_temp_pressure,
                            mean_adsorption=mean_adsorption, std_adsorption=std_adsorption)

    print(adsorption_dataset)
    # 数据集长度
    print(f"Dataset size: {len(adsorption_dataset)}")

    # 获取第一个样本
    # sample = adsorption_dataset[10]
    for sample in adsorption_dataset:
        print(sample)
        print("Graph Data:", sample)
        print("Temperature and Pressure:", sample.temp_pressure)
        print("Adsorption Capacity (label):", sample.y)
