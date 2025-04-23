import os
import re
import numpy as np
import torch
from sympy import false
from torch_geometric.data import Data
from torch.utils.data import Dataset
import pandas as pd
from pymatgen.io.cif import CifParser

# 清理字符串并转换浮点数
def clean_float(value):
    """ 处理形如 '13.6770(0)' 的字符串，将其转换为浮点数 """
    clean_value = re.sub(r'\(.*\)', '', value)
    return float(clean_value)

# 从CIF文件读取原子坐标和晶胞参数
from pymatgen.core import Structure

def read_cif(file_path, occupancy_tolerance=10.0):
    """
    使用 pymatgen 来解析 CIF 文件并返回:
      - atomic_species: [str, ...]  (如 ['Si', 'O', 'O', ...])
      - frac_coords: ndarray shape (N, 3)
      - cart_coords: ndarray shape (N, 3)
      - a, b, c: 晶胞的 a,b,c
      - alpha, beta, gamma: 晶胞角度(单位°)
    """
    # 用 CifParser 并传入 occupancy_tolerance
    parser = CifParser(file_path, occupancy_tolerance=occupancy_tolerance)

    structures = parser.get_structures()  # 自动解析 CIF, 处理对称操作
    structure = structures[0]

    # 从 structure 里取所有原子的化学符号 & 坐标
    atomic_species = []
    frac_coords = []
    cart_coords = []

    for site in structure.sites:
        # site.species_string 可能是 'Si', 'O', 'O2-', 'Na1+' 等
        # 如果只想要元素本身，可用 site.specie.symbol 或 site.species.elements[0].symbol
        # 这里为了匹配你的 atom_properties 里的键名，建议只拿元素符号(不含电荷)
        # 若结构中有混合占位或标注了电荷，需要再做映射。
        element_symbol = list(site.species.keys())[0].symbol
        atomic_species.append(element_symbol)
        frac_coords.append(site.frac_coords)   # 分数坐标
        cart_coords.append(site.coords)        # 笛卡尔坐标

    frac_coords = np.array(frac_coords)
    cart_coords = np.array(cart_coords)

    # 获取晶胞参数
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

    return atomic_species, frac_coords, cart_coords, a, b, c, alpha, beta, gamma


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
    # 用 pymatgen 读取并解析 CIF
    (atomic_species, frac_coords, cartesian_coords,
     a, b, c, alpha, beta, gamma) = read_cif(file_path)

    # 构造节点特征
    node_features = []
    for species in atomic_species:
        # 若在 atom_properties 里没定义该元素，需要做一些默认处理 or 忽略
        if species not in species_to_onehot:
            # 如果你的 species_to_onehot 不包含此元素，可以做一个默认处理
            # 也可以 raise Error，看你自己需求
            print(f"Warning: {species} not in species_to_onehot! Using dummy one-hot.")
            onehot = np.zeros(len(species_to_onehot), dtype=float)
            mass = 0.0
            electrons = 0.0
            valence = 0.0
        else:
            onehot = species_to_onehot[species]
            # 同样 atom_properties 若不存在也要处理一下
            if species not in atom_properties:
                mass = 0.0
                electrons = 0.0
                valence = 0.0
            else:
                mass = atom_properties[species]['mass']
                electrons = atom_properties[species]['electrons']
                valence = atom_properties[species]['valence']

        node_feature = np.concatenate([onehot, [mass, electrons, valence]])
        node_features.append(node_feature)

    node_features = np.array(node_features, dtype=float)

    # 基于范德华半径的成键判断
    edge_index = []
    num_atoms = len(cartesian_coords)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(cartesian_coords[i] - cartesian_coords[j])

            radius_i = get_vdw_radius(atomic_species[i])
            radius_j = get_vdw_radius(atomic_species[j])
            bond_threshold = radius_i/1.5 + radius_j/1.5

            if dist < bond_threshold:
                edge_index.append([i, j])

    edge_index = np.array(edge_index).T

    # 构造 PyG Data
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    return data


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
    def __init__(self, csv_file, cif_directory, mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption, Temp_lambda, pressure_lambda, adsorption_lambda, test=False, handle=False):
        self.data_frame = pd.read_excel(csv_file)
        one_hot_encoded = pd.get_dummies(self.data_frame['adsorbate'], prefix='adsorbate').astype(int)
        one_hot_encoded_array = one_hot_encoded.to_numpy()
        self.data_frame['adsorbate'] = list(one_hot_encoded_array)
        self.cif_directory = cif_directory

        self.test = test
        self.handle = handle

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
        temp = row['temperature']
        pressure = row['Pressure (bar)']
        adsorption = row['Adsorption (mmol/g)']
        zeolite_type = row['zeolite_type']
        if self.test == false:
            type = row['adsorbate']#[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#row['adsorbate']
        else:type = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # print(adsorption)
        # print(type)

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

        if self.handle == True:
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
            adsorptions = (adsorption_boxcox - self.mean_adsorption) / self.std_adsorption


        else:
            temp_pressure = torch.tensor([temp, pressure], dtype=torch.float)
            temp_pressure = (temp_pressure - self.mean_temp_pressure) / self.std_temp_pressure
            adsorptions = (adsorption - self.mean_adsorption) / self.std_adsorption
        type_tensor = torch.tensor(type, dtype=torch.float)
        temp_pressure = torch.cat((temp_pressure, type_tensor))
        # print(temp_pressure)
        # 归一化 adsorption


        # 将归一化后的温度、压力和吸附量添加到 Data 对象中
        graph_data.temp_pressure = temp_pressure
        graph_data.y = torch.tensor(adsorptions, dtype=torch.float)

        return graph_data


def calculate_normalization_params(data_frame,Temp_lambda, pressure_lambda, adsorption_lambda, handle = True):
    temps = data_frame['temperature'].values
    pressures = data_frame['Pressure (bar)'].values
    # types = data_frame['Type'].values
    adsorptions = data_frame['Adsorption (mmol/g)'].values
    if handle == True:
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
    else:
        temp_boxcox, pressure_boxcox,adsorption_boxcox = temps,pressures, adsorptions
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
            atomic_species, frac_coords, cartesian_coords, a, b, c, alpha, beta, gamma = read_cif(file_path)
            species_set.update(atomic_species)
    print('原子类型', sorted(list(species_set)))
    return sorted(list(species_set))

if __name__ == '__main__':
    csv_file = 'newdata/nosmooth/database.xlsx'  # 包含温度、压力、吸附量和沸石种类的 Excel 文件
    cif_directory = './cif_file/'  # CIF 文件的目录
    Temp_lambda, pressure_lambda, adsorption_lambda = 2.197387215938357, 0.09754163115529348, 0.0829811630928598 #2.197398896269478, 0.11636148040038101, 0.12171020993011404
    mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption = calculate_normalization_params(
        pd.read_excel(csv_file), Temp_lambda, pressure_lambda, adsorption_lambda,handle = False)
    print(mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption)
    # 创建数据集实例
    adsorption_dataset = AdsorptionDataset(csv_file=csv_file, cif_directory=cif_directory,
                            mean_temp_pressure=mean_temp_pressure, std_temp_pressure=std_temp_pressure,
                            mean_adsorption=mean_adsorption, std_adsorption=std_adsorption, Temp_lambda=Temp_lambda, pressure_lambda=pressure_lambda, adsorption_lambda=adsorption_lambda, handle = False)

    print(adsorption_dataset)
    # 数据集长度
    print(f"Dataset size: {len(adsorption_dataset)}")

    # 获取第一个样本
    sample = adsorption_dataset[0]
    # for sample in adsorption_dataset:
    print(sample)
    print("Graph Data:", sample.edge_index)
    print("Temperature and Pressure:", sample.temp_pressure)
    print("Adsorption Capacity (label):", sample.y)