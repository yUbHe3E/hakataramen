import os
import re
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import pandas as pd

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

# 定义原子特性，包括 O₂⁻(H₂O) 作为独立物种的质量和电子结构信息
atom_properties = {
    'Al': {'mass': 26.98, 'electrons': 13, 'valence': 3},
    'C': {'mass': 12.01, 'electrons': 6, 'valence': 4},
    'N': {'mass': 14.01, 'electrons': 7, 'valence': 5},
    'O': {'mass': 16.00, 'electrons': 8, 'valence': 6},
    'P': {'mass': 30.97, 'electrons': 15, 'valence': 5},
    'Si': {'mass': 28.09, 'electrons': 14, 'valence': 4},
    'O2-(H2O)': {'mass': 18.015 + 16.00, 'electrons': 10, 'valence': 8},  # 水+氧
    'K': {'mass': 39.10, 'electrons': 19, 'valence': 1},  # 钾元素
    'Na': {'mass': 22.99, 'electrons': 11, 'valence': 1},  # 钠元素
    'Li': {'mass': 6.94, 'electrons': 3, 'valence': 1},    # 锂元素
}

van_der_waals_radius = {
    'Al': 1.84,  # 铝
    'C': 1.70,   # 碳
    'N': 1.55,   # 氮
    'O': 1.52,   # 氧
    'O2-(H2O)': 1.52,  # 基于氧原子的范德华半径
    'P': 1.80,   # 磷
    'Si': 2.10,  # 硅
    'K': 2.75,   # 钾元素
    'Na': 2.27,  # 钠元素
    'Li': 1.82   # 锂元素
}

def get_vdw_radius(atom):
    """ 根据原子符号返回范德华半径 """
    return van_der_waals_radius.get(atom, 1.7)  # 如果原子类型未定义，使用默认值 1.7

# 处理单个CIF文件并生成图数据
def process_cif_file(file_path, species_to_onehot):
    atomic_species, fractional_coords, a, b, c = read_cif(file_path)

    # 将分数坐标转换为笛卡尔坐标
    cartesian_coords = np.array([[a * x, b * y, c * z] for x, y, z in fractional_coords])

    # 生成节点特征，包含原子种类、质量、电子结构等信息
    node_features = []
    for species in atomic_species:
        onehot = species_to_onehot[species]  # 原子种类的 one-hot 编码
        mass = atom_properties[species]['mass']  # 原子质量
        electrons = atom_properties[species]['electrons']  # 原子电子数
        valence = atom_properties[species]['valence']  # 原子价电子数
        # 将所有特征拼接成节点特征
        node_feature = np.concatenate([onehot, [mass, electrons, valence]])
        node_features.append(node_feature)
    # print(node_features)
    node_features = np.array(node_features)

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

    # 转换为 PyTorch tensor 格式
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # 创建图数据
    return Data(x=node_features, edge_index=edge_index)

# 动态构建原子类型集
def build_unique_species(directory):
    species_set = set()
    for filename in os.listdir(directory):
        if filename.endswith(".cif") or filename.endswith(".cif_"):
            file_path = os.path.join(directory, filename)
            atomic_species, _, _, _, _ = read_cif(file_path)
            species_set.update(atomic_species)
    return sorted(list(species_set))



# 自定义 PyTorch 数据集类
class AdsorptionDataset(Dataset):
    def __init__(self, csv_file, cif_directory, mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption, Temp_lambda, pressure_lambda, adsorption_lambda):
        self.data_frame = pd.read_excel(csv_file)
        self.cif_directory = cif_directory

        self.mean_temp_pressure = mean_temp_pressure
        self.std_temp_pressure = std_temp_pressure
        self.mean_adsorption = mean_adsorption
        self.std_adsorption = std_adsorption

        self.Temp_lambda = Temp_lambda
        self.pressure_lambda = pressure_lambda
        self.adsorption_lambda = adsorption_lambda

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
        # type = row['Type']

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

        # 对吸附量进行 Box-Cox 变换
        if self.Temp_lambda != 0:
            temp_boxcox = (np.power(temp+ 1e-20, self.Temp_lambda) - 1) / self.Temp_lambda
        else:
            temp_boxcox = np.log(temp+ 1e-20)

        if self.pressure_lambda != 0:
            pressure_boxcox = (np.power(pressure+ 1e-20, self.pressure_lambda) - 1) / self.pressure_lambda
        else:
            pressure_boxcox = np.log(pressure+ 1e-20)

        # 对吸附量进行 Box-Cox 变换
        if self.adsorption_lambda != 0:
            adsorption_boxcox = (np.power(adsorption+ 1e-20, self.adsorption_lambda) - 1) / self.adsorption_lambda
        else:
            adsorption_boxcox = np.log(adsorption+ 1e-20)

        # 归一化温度、压力和type
        temp_pressure = torch.tensor([temp_boxcox, pressure_boxcox], dtype=torch.float)#, type
        temp_pressure = (temp_pressure - self.mean_temp_pressure) / self.std_temp_pressure

        # 归一化 adsorption
        normalized_adsorption = (adsorption_boxcox - self.mean_adsorption) / self.std_adsorption

        # 将归一化后的温度、压力和吸附量添加到 Data 对象中
        graph_data.temp_pressure = temp_pressure
        graph_data.y = torch.tensor(normalized_adsorption, dtype=torch.float)

        return graph_data


def calculate_normalization_params(data_frame,Temp_lambda, pressure_lambda, adsorption_lambda):
    temps = data_frame['Temp'].values
    pressures = data_frame['Pressure(Bar)'].values
    types = data_frame['Type'].values
    adsorptions = data_frame['total_adsorption(mmol/g)'].values
    # 对吸附量进行 Box-Cox 变换
    if Temp_lambda != 0:
        temp_boxcox = (np.power(temps+ 1e-20, Temp_lambda) - 1) / Temp_lambda
    else:
        temp_boxcox = np.log(temps+ 1e-20)

    if pressure_lambda != 0:
        pressure_boxcox = (np.power(pressures+ 1e-20, pressure_lambda) - 1) / pressure_lambda
    else:
        pressure_boxcox = np.log(pressures+ 1e-20)

    # 对吸附量进行 Box-Cox 变换
    if adsorption_lambda != 0:
        adsorption_boxcox = (np.power(adsorptions+ 1e-20, adsorption_lambda) - 1) / adsorption_lambda
    else:
        adsorption_boxcox = np.log(adsorptions+ 1e-20)
    # 计算温度、压力、type 和 adsorption 的均值和标准差
    mean_temp_pressure = np.mean([temp_boxcox, pressure_boxcox], axis=1)#, types
    std_temp_pressure = np.std([temp_boxcox, pressure_boxcox], axis=1)#, types
    mean_adsorption = np.mean(adsorption_boxcox)
    std_adsorption = np.std(adsorption_boxcox)

    return mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption

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

if __name__ == '__main__':
    csv_file = 'database.xlsx'  # 包含温度、压力、吸附量和沸石种类的 Excel 文件
    cif_directory = './cif_file/'  # CIF 文件的目录
    Temp_lambda, pressure_lambda, adsorption_lambda = -3.111674466482276, 0.12938587453917788, 0.2782773034110868
    mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption = calculate_normalization_params(
        pd.read_excel(csv_file), Temp_lambda, pressure_lambda, adsorption_lambda)
    print(mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption)
    # 创建数据集实例
    adsorption_dataset = AdsorptionDataset(csv_file=csv_file, cif_directory=cif_directory,
                            mean_temp_pressure=mean_temp_pressure, std_temp_pressure=std_temp_pressure,
                            mean_adsorption=mean_adsorption, std_adsorption=std_adsorption, Temp_lambda=Temp_lambda, pressure_lambda=pressure_lambda, adsorption_lambda=adsorption_lambda)

    print(adsorption_dataset)
    # 数据集长度
    print(f"Dataset size: {len(adsorption_dataset)}")

    # 获取第一个样本
    sample = adsorption_dataset[500]
    # for sample in adsorption_dataset:
    print(sample)
    print("Graph Data:", sample.edge_index)
    print("Temperature and Pressure:", sample.temp_pressure)
    print("Adsorption Capacity (label):", sample.y)