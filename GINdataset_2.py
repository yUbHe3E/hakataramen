import os
import re
import numpy as np
import torch
from sympy import false
from torch_geometric.data import Data
from torch.utils.data import Dataset
import pandas as pd

# ============ 新增：pymatgen 相关 =============
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN  # 或其他近邻算法
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



from pymatgen.io.cif import CifParser

def read_cif_with_pymatgen(file_path, occupancy_tolerance=10.0):
    """
    使用 pymatgen.io.cif.CifParser 从给定的 CIF 文件解析得到:
      1) atomic_species: 原子符号列表 (如 ['Si', 'O', 'O', ...])
      2) cartesian_coords: 笛卡尔坐标数组 shape=(N,3)
      3) structure: pymatgen 的 Structure 对象

    其中 occupancy_tolerance=1.2 表示若某些位点总占据值>1.0但小于1.2,
    pymatgen 会自动做 rescale/修正, 避免解析失败.
    """

    # 用 CifParser 并传入 occupancy_tolerance
    parser = CifParser(file_path, occupancy_tolerance=occupancy_tolerance)

    # 一般 CIF 只包含一个 data block, 所以只取第一个结构
    # 如果有多个, 视需求再做处理
    structures = parser.get_structures()
    # if len(structures) == 0:
        # raise ValueError(f"Failed to parse any structure from {file_path}")

    structure = structures[0]

    atomic_species = []
    cart_coords = []
    for site in structure.sites:
        # site.specie.symbol 通常给出元素符号, e.g. 'O','Si','Na'
        # 如果有 'O2-' 之类需要你自行映射到 'O'
        symbol = site.specie.symbol
        atomic_species.append(symbol)
        cart_coords.append(site.coords)

    cartesian_coords = np.array(cart_coords)
    return atomic_species, cartesian_coords, structure

# 近邻判断函数：利用CrystalNN找邻居
def get_edges_pmg(structure):
    """
    给定pymatgen的Structure，使用CrystalNN来找出所有原子对(i, j)。
    返回无向边列表: [(i, j), ...], 其中 i<j
    """
    cnn = CrystalNN()
    edges = []
    num_sites = len(structure)

    for i in range(num_sites):
        nn_info = cnn.get_nn_info(structure, i)
        for neighbor in nn_info:
            j = neighbor['site_index']
            if j > i:  # 避免重复
                edges.append((i, j))

    return edges

# ============ 保留你原先的 atom_properties 和 van_der_waals_radius 等 ============
atom_properties = {
    'Al': {'mass': 26.98, 'electrons': 13, 'valence': 3},
    'C': {'mass': 12.01, 'electrons': 6, 'valence': 4},
    'N': {'mass': 14.01, 'electrons': 7, 'valence': 5},
    'O': {'mass': 16.00, 'electrons': 8, 'valence': 6},
    'P': {'mass': 30.97, 'electrons': 15, 'valence': 5},
    'Si': {'mass': 28.09, 'electrons': 14, 'valence': 4},
    'O2-(H2O)': {'mass': 18.015 + 16.00, 'electrons': 10, 'valence': 8},  # 水+氧
    'K': {'mass': 39.10, 'electrons': 19, 'valence': 1},
    'Na': {'mass': 22.99, 'electrons': 11, 'valence': 1},
    'Li': {'mass': 6.94, 'electrons': 3, 'valence': 1},
}

van_der_waals_radius = {
    'Al': 1.84,
    'C':  1.70,
    'N':  1.55,
    'O':  1.52,
    'O2-(H2O)': 1.52,
    'P':  1.80,
    'Si': 2.10,
    'K':  2.75,
    'Na': 2.27,
    'Li': 1.82
}

def get_vdw_radius(atom):
    """ 根据原子符号返回范德华半径 """
    return van_der_waals_radius.get(atom, 1.7)

# ============ 新的 process_cif_file，改用 pymatgen & CrystalNN ============
def process_cif_file(file_path, species_to_onehot):
    """
    1) 解析CIF -> pymatgen Structure
    2) 拿到完整的 atomic_species、cartesian_coords
    3) 用 CrystalNN 获取邻接关系
    4) 构建 PyTorch Geometric Data
    """
    atomic_species, cartesian_coords, structure = read_cif_with_pymatgen(file_path)

    # 构建节点特征
    node_features = []
    for species in atomic_species:
        # 如果遇到 pymatgen 里出现 'O2-' 等，请手动映射到 'O'
        # 这里假设你没遇到或不在意
        if species not in species_to_onehot:
            # 若不在 species_to_onehot，则默认0向量
            onehot = np.zeros(len(species_to_onehot))
        else:
            onehot = species_to_onehot[species]

        if species not in atom_properties:
            mass = electrons = valence = 0.0
        else:
            mass      = atom_properties[species]['mass']
            electrons = atom_properties[species]['electrons']
            valence   = atom_properties[species]['valence']

        node_feature = np.concatenate([onehot, [mass, electrons, valence]])
        node_features.append(node_feature)

    node_features = np.array(node_features, dtype=float)

    # 构建edge_index
    edges = get_edges_pmg(structure)  # list of (i, j)
    edges_array = np.array(edges).T if len(edges) > 0 else np.zeros((2,0), dtype=int)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges_array, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data

# ============ 保留其余与 Dataset/DataFrame 相关的代码基本不变 ============

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

        # 用新的方法, 其实 build_unique_species 也可改用 pymatgen, 这里先不动
        self.unique_species = build_unique_species(cif_directory)
        self.species_to_onehot = {
            species: np.eye(len(self.unique_species))[i] for i, species in enumerate(self.unique_species)
        }

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        temp = row['temperature']
        pressure = row['Pressure (bar)']
        adsorption = row['Adsorption (mmol/g)']
        zeolite_type = row['zeolite_type']
        if self.test == false:
            type = row['adsorbate']
        else:
            type = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # 匹配CIF文件
        cif_file = None
        for filename in os.listdir(self.cif_directory):
            if filename.startswith(zeolite_type) and filename.endswith(".cif_"):
                cif_file = os.path.join(self.cif_directory, filename)
                break

        if cif_file is None:
            raise FileNotFoundError(f"No CIF file found for zeolite type: {zeolite_type}")

        # 使用新的 process_cif_file
        graph_data = process_cif_file(cif_file, self.species_to_onehot)

        # 后续的温度、压力、吸附量的处理不变
        if self.handle == True:
            if self.Temp_lambda != 0:
                temp_boxcox = (np.power(temp+ 1e-20, self.Temp_lambda) - 1) / self.Temp_lambda
            else:
                temp_boxcox = np.log(temp+ 1e-20)

            if self.pressure_lambda != 0:
                pressure_boxcox = (np.power(pressure+ 1e-20, self.pressure_lambda) - 1) / self.pressure_lambda
            else:
                pressure_boxcox = np.log(pressure+ 1e-20)

            if self.adsorption_lambda != 0:
                adsorption_boxcox = (np.power(adsorption+ 1e-20, self.adsorption_lambda) - 1) / self.adsorption_lambda
            else:
                adsorption_boxcox = np.log(adsorption+ 1e-20)

            temp_pressure = torch.tensor([temp_boxcox, pressure_boxcox], dtype=torch.float)
            temp_pressure = (temp_pressure - self.mean_temp_pressure) / self.std_temp_pressure
            adsorptions = (adsorption_boxcox - self.mean_adsorption) / self.std_adsorption
        else:
            temp_pressure = torch.tensor([temp, pressure], dtype=torch.float)
            temp_pressure = (temp_pressure - self.mean_temp_pressure) / self.std_temp_pressure
            adsorptions = (adsorption - self.mean_adsorption) / self.std_adsorption

        type_tensor = torch.tensor(type, dtype=torch.float)
        temp_pressure = torch.cat((temp_pressure, type_tensor))

        graph_data.temp_pressure = temp_pressure
        graph_data.y = torch.tensor(adsorptions, dtype=torch.float)

        return graph_data

# 其余函数(如 clean_float, build_unique_species, 等)可以保留或删除
# 因为你已经不再需要手写 parse CIF, 只要 build_unique_species 用的话，需要只返回 species。
def clean_float(value):
    clean_value = re.sub(r'\(.*\)', '', value)
    return float(clean_value)


def build_unique_species(directory, occupancy_tolerance=10.0):
    """
    使用 pymatgen 的 CifParser 并指定 occupancy_tolerance，以获取
    目录下所有 .cif/.cif_ 文件里出现的原子符号。
    """
    species_set = set()

    for filename in os.listdir(directory):
        if filename.endswith(".cif") or filename.endswith(".cif_"):
            file_path = os.path.join(directory, filename)
            try:
                # 1) 用 CifParser 并传入 occupancy_tolerance
                parser = CifParser(file_path, occupancy_tolerance=occupancy_tolerance)
                # 2) 拿到解析后的所有 structure (有时一个 CIF 里会有多个 data block)
                structures = parser.get_structures()

                if not structures:
                    # 如果没有解析到结构，则跳过
                    print(f"Warning: No structure parsed in {file_path} after tolerance={occupancy_tolerance}.")
                    continue

                # 一般只取第 0 个 structure
                structure = structures[0]

                # 3) 收集所有 site 里的 specie.symbol
                for site in structure.sites:
                    symbol = site.specie.symbol
                    # 这里可以做简易映射，比如:
                    # if 'O' in symbol: symbol = 'O'
                    # if 'Na' in symbol: symbol = 'Na'
                    # ...
                    species_set.add(symbol)

            except Exception as e:
                print(f"Warning: Failed to parse {file_path} with error: {e}")
                # 解析失败就跳过
                continue


    species_list = sorted(species_set)
    print(f"Unique species found via pymatgen (tolerance={occupancy_tolerance}): {species_list}")
    return species_list

# ======================================================================
# 下面是原先 main 函数（包含 calculate_normalization_params 等）
# 仅示例如何跑整个流程
def calculate_normalization_params(data_frame, Temp_lambda, pressure_lambda, adsorption_lambda, handle = True):
    temps = data_frame['temperature'].values
    pressures = data_frame['Pressure (bar)'].values
    adsorptions = data_frame['Adsorption (mmol/g)'].values

    if handle:
        if Temp_lambda != 0:
            temp_boxcox = (np.power(temps+ 1e-20, Temp_lambda) - 1) / Temp_lambda
        else:
            temp_boxcox = np.log(temps+ 1e-20)

        if pressure_lambda != 0:
            pressure_boxcox = (np.power(pressures+ 1e-20, pressure_lambda) - 1) / pressure_lambda
        else:
            pressure_boxcox = np.log(pressures+ 1e-20)

        if adsorption_lambda != 0:
            adsorption_boxcox = (np.power(adsorptions+ 1e-20, adsorption_lambda) - 1) / adsorption_lambda
        else:
            adsorption_boxcox = np.log(adsorptions+ 1e-20)
    else:
        temp_boxcox, pressure_boxcox, adsorption_boxcox = temps, pressures, adsorptions

    mean_temp_pressure = np.mean([temp_boxcox, pressure_boxcox], axis=1)
    std_temp_pressure = np.std([temp_boxcox, pressure_boxcox], axis=1)
    mean_adsorption = np.mean(adsorption_boxcox)
    std_adsorption = np.std(adsorption_boxcox)

    return mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption

if __name__ == '__main__':
    csv_file = 'newdata/nosmooth/database.xlsx'
    cif_directory = './cif_file/'
    Temp_lambda, pressure_lambda, adsorption_lambda = (2.197387215938357, 0.09754163115529348, 0.0829811630928598)
    mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption = calculate_normalization_params(
        pd.read_excel(csv_file), Temp_lambda, pressure_lambda, adsorption_lambda, handle = False)

    print(mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption)

    adsorption_dataset = AdsorptionDataset(
        csv_file=csv_file,
        cif_directory=cif_directory,
        mean_temp_pressure=mean_temp_pressure,
        std_temp_pressure=std_temp_pressure,
        mean_adsorption=mean_adsorption,
        std_adsorption=std_adsorption,
        Temp_lambda=Temp_lambda,
        pressure_lambda=pressure_lambda,
        adsorption_lambda=adsorption_lambda,
        handle=False
    )

    print(adsorption_dataset)
    print(f"Dataset size: {len(adsorption_dataset)}")

    sample = adsorption_dataset[0]
    print(sample)
    print("Graph Data (edge_index):", sample.edge_index)
    print("Temperature + Pressure + OneHot(adsorbate):", sample.temp_pressure)
    print("Adsorption Capacity (label):", sample.y)
