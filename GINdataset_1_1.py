import os
import copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from pymatgen.io.cif import CifParser

###################################################
# 1) CIF 解析 & 构图工具
###################################################

atom_properties = {
    'Al': {'mass': 26.98, 'electrons': 13, 'valence': 3},
    'C': {'mass': 12.01, 'electrons': 6, 'valence': 4},
    'N': {'mass': 14.01, 'electrons': 7, 'valence': 5},
    'O': {'mass': 16.00, 'electrons': 8, 'valence': 6},
    'P': {'mass': 30.97, 'electrons': 15, 'valence': 5},
    'Si': {'mass': 28.09, 'electrons': 14, 'valence': 4},
    'O2-(H2O)': {'mass': 18.015 + 16.00, 'electrons': 10, 'valence': 8},
    'K': {'mass': 39.10, 'electrons': 19, 'valence': 1},
    'Na': {'mass': 22.99, 'electrons': 11, 'valence': 1},
    'Li': {'mass': 6.94, 'electrons': 3, 'valence': 1},
}

van_der_waals_radius = {
    'Al': 1.84,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'O2-(H2O)': 1.52,
    'P': 1.80,
    'Si': 2.10,
    'K': 2.75,
    'Na': 2.27,
    'Li': 1.82,
}


def get_vdw_radius(atom):
    return van_der_waals_radius.get(atom, 1.7)


def boxcox_transform(value, lam):
    """对单个标量做 Box-Cox 变换，避免 0 或负数报错"""
    value = value + 1e-20
    if lam != 0:
        return (value ** lam - 1) / lam
    else:
        return np.log(value)


def read_cif(file_path, occupancy_tolerance=10.0):
    """
    使用 pymatgen 解析 CIF 文件，并返回:
      - atomic_species: [str, ...]
      - frac_coords: ndarray (N, 3)
      - cart_coords: ndarray (N, 3)
      - a, b, c, alpha, beta, gamma
    """
    parser = CifParser(file_path, occupancy_tolerance=occupancy_tolerance)
    structures = parser.get_structures()
    structure = structures[0]

    atomic_species = []
    frac_coords = []
    cart_coords = []
    for site in structure.sites:
        element_symbol = list(site.species.keys())[0].symbol
        atomic_species.append(element_symbol)
        frac_coords.append(site.frac_coords)
        cart_coords.append(site.coords)

    frac_coords = np.array(frac_coords)
    cart_coords = np.array(cart_coords)

    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

    return atomic_species, frac_coords, cart_coords, a, b, c, alpha, beta, gamma


def process_cif_file(file_path, species_to_onehot):
    """
    解析单个 .cif 文件，并返回 PyG Data(x, edge_index)。
    不包含 temp_pressure / y，因为它们是和 CSV 行挂钩的。
    """
    (atomic_species, frac_coords, cart_coords,
     a, b, c, alpha, beta, gamma) = read_cif(file_path)

    # 构造节点特征
    node_features = []
    for species in atomic_species:
        if species in species_to_onehot:
            onehot = species_to_onehot[species]
        else:
            # 如果出现了新的原子种类，未在 species_to_onehot 里，就给个0
            print(f"[Warning] {species} not in species_to_onehot, using dummy one-hot.")
            onehot = np.zeros(len(species_to_onehot))

        if species in atom_properties:
            mass = atom_properties[species]['mass']
            electrons = atom_properties[species]['electrons']
            valence = atom_properties[species]['valence']
        else:
            mass = 0.0
            electrons = 0.0
            valence = 0.0

        node_feature = np.concatenate([onehot, [mass, electrons, valence]])
        node_features.append(node_feature)

    x = torch.tensor(node_features, dtype=torch.float)

    # 基于范德华半径的简单连边
    edge_index = []
    num_atoms = len(cart_coords)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(cart_coords[i] - cart_coords[j])
            r_i = get_vdw_radius(atomic_species[i])
            r_j = get_vdw_radius(atomic_species[j])
            if dist < (r_i + r_j):
                edge_index.append([i, j])

    edge_index = np.array(edge_index).T  # (2, E)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data


###################################################
# 2) 构建 {zeolite_type -> PyG Data} 的字典
###################################################
def build_unique_species(cif_directory):
    """扫描目录里的所有 .cif / .cif_，收集原子种类"""
    species_set = set()
    for fn in os.listdir(cif_directory):
        if fn.endswith(".cif") or fn.endswith(".cif_"):
            fpath = os.path.join(cif_directory, fn)
            atomic_species, frac_coords, cart_coords, a, b, c, alpha, beta, gamma = read_cif(fpath)
            species_set.update(atomic_species)
    return sorted(list(species_set))


def prepare_cif_dict(cif_directory, cif_dict_path='graphs_dict.pt'):
    """只解析一次 .cif，构建 {zeolite_type -> Data} 并保存到 cif_dict_path"""
    # 1) 先收集所有物种，给它们做 one-hot
    unique_sp = build_unique_species(cif_directory)
    species_to_onehot = {sp: np.eye(len(unique_sp))[i] for i, sp in enumerate(unique_sp)}

    # 2) 构建 {zeolite_type -> graph_data}
    cif_dict = {}
    for fn in os.listdir(cif_directory):
        if fn.endswith(".cif") or fn.endswith(".cif_"):
            # zeolite_type 可以用文件名前缀
            # 例如 "FAU_11.cif_" -> "FAU_11"
            zeotype = fn.replace('.cif_', '')
            path = os.path.join(cif_directory, fn)
            data = process_cif_file(path, species_to_onehot)
            cif_dict[zeotype] = data

    # 3) 保存
    print('cif_dict',cif_dict)
    print('unique_sp',unique_sp)
    torch.save((cif_dict, unique_sp), cif_dict_path)
    print(f"[INFO] {len(cif_dict)} CIF files parsed, saved to '{cif_dict_path}'.")


###################################################
# 3) 对 adsorbate 做 one-hot
###################################################
def prepare_adsorbate_one_hot(csv_file, adsorbate_map_path='adsorbate_map.pt'):
    """
    读取 CSV 里所有 adsorbate（如 'CO2'/'N2'等），用 pd.get_dummies 或手写映射，
    并把“{adsorbate_str -> one-hot向量}”保存到 adsorbate_map_path。
    """
    df = pd.read_excel(csv_file)
    # 利用 get_dummies 做映射
    # 这样 "adsorbate" 列会变成多列 'adsorbate_CO2', 'adsorbate_N2' ...
    # 也可以自己手写: collect unique adsorbates, then map each to an index.
    # 这里用 get_dummies 后，每行就是一行 one-hot:
    dummies = pd.get_dummies(df['adsorbate'], prefix='adsorbate').astype(int)
    # dummies.columns 例如: ['adsorbate_CO2', 'adsorbate_N2', ...]

    # 建立 “adsorbate字符串 -> one-hot向量” 的dict
    adsorbate_to_vec = {}
    for i, cat in enumerate(dummies.columns):
        # cat 形如 'adsorbate_CO2'
        # 也可以 parse 下 '_CO2'
        ads_name = cat.split('_', 1)[1]  # 拆出 CO2
        # 构造一个单位矩阵 row[i], i.e. e_i
        one_hot = np.zeros(len(dummies.columns))
        one_hot[i] = 1
        adsorbate_to_vec[ads_name] = one_hot

    torch.save(adsorbate_to_vec, adsorbate_map_path)
    print(f"[INFO] Adsorbate one-hot map saved to '{adsorbate_map_path}', total={len(adsorbate_to_vec)}.")


###################################################
# 4) Dataset：加载缓存，并在每行拼上 temp_pressure & y
###################################################
class AdsorptionDataset(Dataset):
    def __init__(
            self,
            csv_file,
            cif_directory,
            mean_temp_pressure,
            std_temp_pressure,
            mean_adsorption,
            std_adsorption,
            Temp_lambda,
            pressure_lambda,
            adsorption_lambda,
            test=False,
            handle=False,
            cif_dict_path='graphs_dict.pt',
            adsorbate_map_path='adsorbate_map.pt',
            idx_mask=None
    ):
        self.df = pd.read_excel(csv_file)
        self.test = test
        self.handle = handle
        if idx_mask is not None:  # ★ 新增：按布尔索引过滤
            self.df = self.df[idx_mask].reset_index(drop=True)
        self.mean_temp_pressure = mean_temp_pressure
        self.std_temp_pressure = std_temp_pressure
        self.mean_adsorption = mean_adsorption
        self.std_adsorption = std_adsorption
        self.Temp_lambda = Temp_lambda
        self.pressure_lambda = pressure_lambda
        self.adsorption_lambda = adsorption_lambda

        # 1) 如果没有解析过 .cif，就先建 cache
        if not os.path.exists(cif_dict_path):
            print(f"[INFO] '{cif_dict_path}' not found, building it...")
            prepare_cif_dict(cif_directory, cif_dict_path)
        # 2) 加载 {zeolite_type -> Data} 字典
        cif_info = torch.load(cif_dict_path, weights_only=False)
        self.cif_dict = cif_info[0]  # dict
        self.unique_sp = cif_info[1]  # 可能用不到

        # 3) 如果没有解析过 adsorbate，就先做 get_dummies 一次
        if not os.path.exists(adsorbate_map_path):
            print(f"[INFO] '{adsorbate_map_path}' not found, building it...")
            prepare_adsorbate_one_hot(csv_file, adsorbate_map_path)
        # 4) 加载 {adsorbate_str -> one-hot向量} 字典
        self.ads_map = torch.load(adsorbate_map_path, weights_only=False)

        print(f"[INFO] Loaded {len(self.cif_dict)} CIF graphs & {len(self.ads_map)} adsorbates.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        zeolite_type = row['zeolite_type']
        if zeolite_type not in self.cif_dict:
            # CSV 里出现了沸石类型，但 .cif 里没有
            raise KeyError(f"[Error] zeolite '{zeolite_type}' not found in cif_dict")

        temp = row['temperature']
        pressure = row['Pressure (bar)']
        adsorption = row['Adsorption (mmol/g)']
        adsorbate_str = row['adsorbate']  # 这是 Excel 里的原始字符串 (CO2, N2, ...)

        # (1) 拿到图
        #    注意: 多条数据会共享同一个 graph_data 对象 => 要复制一份
        #    否则，后面改 temp_pressure / y，会互相影响
        original_graph = self.cif_dict[zeolite_type]
        graph_data = copy.deepcopy(original_graph)

        # (2) 处理 adsorbate 的 one-hot
        #     如果不是 test，就用真实 one-hot；如果是 test，就用你硬编码那种向量
        #     （这里仅演示：也可自定义别的写法）
        if not self.test:
            if adsorbate_str not in self.ads_map:
                # 如果 CSV 新增了某个adsorbate，ads_map里还没有 -> 需要重建/更新
                print(f"[Warning] '{adsorbate_str}' not in adsorbate_map, use zero vector.")
                type_vec = np.zeros(len(self.ads_map))
            else:
                type_vec = self.ads_map[adsorbate_str]
        else:
            # 你在原代码里写的硬编码
            type_vec = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # (3) Box-Cox
        if self.handle:
            temp = boxcox_transform(temp, self.Temp_lambda)
            pressure = boxcox_transform(pressure, self.pressure_lambda)
            adsorption = boxcox_transform(adsorption, self.adsorption_lambda)

        # (4) 归一化
        t_p_tensor = torch.tensor([temp, pressure], dtype=torch.float)
        t_p_tensor = (t_p_tensor - self.mean_temp_pressure) / self.std_temp_pressure
        ads_tensor = (torch.tensor(adsorption, dtype=torch.float) - self.mean_adsorption) / self.std_adsorption

        # (5) 拼接 one-hot
        type_tensor = torch.tensor(type_vec, dtype=torch.float)
        temp_pressure = torch.cat([t_p_tensor, type_tensor], dim=0)  # [2 + len(type_vec)]

        # (6) 塞进 graph_data
        graph_data.temp_pressure = temp_pressure
        graph_data.y = ads_tensor.unsqueeze(-1)  # shape [1], 你也可直接标量

        return graph_data


###################################################
# 5) 计算归一化参数 (跟你以前的逻辑一致)
###################################################
def calculate_normalization_params(df, Temp_lambda, pressure_lambda, adsorption_lambda, handle=True):
    temps = df['temperature'].values
    pressures = df['Pressure (bar)'].values
    adsorptions = df['Adsorption (mmol/g)'].values

    if handle:
        temp_boxcox = np.array([boxcox_transform(t, Temp_lambda) for t in temps], dtype=float)
        press_boxcox = np.array([boxcox_transform(p, pressure_lambda) for p in pressures], dtype=float)
        ads_boxcox = np.array([boxcox_transform(a, adsorption_lambda) for a in adsorptions], dtype=float)
    else:
        temp_boxcox = temps
        press_boxcox = pressures
        ads_boxcox = adsorptions

    mean_temp_pressure = np.mean([temp_boxcox, press_boxcox], axis=1)
    std_temp_pressure = np.std([temp_boxcox, press_boxcox], axis=1)
    mean_adsorption = np.mean(ads_boxcox)
    std_adsorption = np.std(ads_boxcox)

    return mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption


###################################################
# 6) 运行/测试示例
###################################################
if __name__ == '__main__':
    # 你可以修改成真实路径
    csv_file = 'newdata/nosmooth/database.xlsx'
    cif_directory = './cif_file/'
    cif_dict_path = './temp/graphs_dict.pt'
    adsorbate_map_path = './temp/adsorbate_map.pt'

    # Box-Cox 参数
    Temp_lambda, pressure_lambda, adsorption_lambda = 2.197, 0.097, 0.082

    # 先计算归一化参数
    df = pd.read_excel(csv_file)
    mean_tp, std_tp, mean_ads, std_ads = calculate_normalization_params(
        df, Temp_lambda, pressure_lambda, adsorption_lambda, handle=True
    )
    print("[INFO] Normalization parameters:")
    print("  mean_temp_pressure =", mean_tp)
    print("  std_temp_pressure  =", std_tp)
    print("  mean_adsorption    =", mean_ads)
    print("  std_adsorption     =", std_ads)

    # 构建数据集
    dataset = AdsorptionDataset(
        csv_file=csv_file,
        cif_directory=cif_directory,
        mean_temp_pressure=mean_tp,
        std_temp_pressure=std_tp,
        mean_adsorption=mean_ads,
        std_adsorption=std_ads,
        Temp_lambda=Temp_lambda,
        pressure_lambda=pressure_lambda,
        adsorption_lambda=adsorption_lambda,
        test=False,  # or True
        handle=True,
        cif_dict_path=cif_dict_path,
        adsorbate_map_path=adsorbate_map_path
    )

    print("\n[INFO] Dataset length =", len(dataset))

    # 测试取一个样本
    sample = dataset[0]
    print("[INFO] Sample 0 => ", sample)
    print("  sample.x.shape =", sample.x.shape)
    print("  sample.edge_index.shape =", sample.edge_index.shape)
    print("  sample.temp_pressure =", sample.temp_pressure)
    print("  sample.y =", sample.y)

    # 做个 DataLoader 测试
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for i, batch in enumerate(loader):
        print(f"\n[Batch {i}] =>", batch)
        if i >= 1:
            break
