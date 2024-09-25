import os
import numpy as np
import torch
from torch_geometric.data import Data
from CifFile import ReadCif  # PycifRW


# 动态构建原子类型集
def build_unique_species(directory):
    """ 从所有CIF文件中构建统一的原子类型集 """
    species_set = set()

    for filename in os.listdir(directory):
        if filename.endswith(".cif") or filename.endswith(".cif_"):
            file_path = os.path.join(directory, filename)
            try:
                # 使用 PycifRW 读取CIF文件
                cif_file = ReadCif(file_path)

                # 通常第一个CIF block包含所需的结构信息
                cif_block = cif_file.first_block()

                # 提取原子类型
                atomic_species = cif_block['_atom_site_type_symbol']
                species_set.update(atomic_species)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    return sorted(list(species_set))


# 处理单个CIF文件
def process_cif_file(file_path, species_to_onehot):
    """ 处理单个CIF文件，返回节点特征和边索引 """
    try:
        # 使用 PycifRW 读取CIF文件
        cif_file = ReadCif(file_path)
        cif_block = cif_file.first_block()  # 获取第一个 block，通常包含结构信息

        # 提取原子类型
        atomic_species = list(cif_block['_atom_site_type_symbol'])

        # 提取分数坐标
        fractional_coords = np.array([
            list(map(float, [cif_block['_atom_site_fract_x'][i],
                             cif_block['_atom_site_fract_y'][i],
                             cif_block['_atom_site_fract_z'][i]]))
            for i in range(len(atomic_species))
        ])

        # 提取晶胞参数
        a = float(cif_block['_cell_length_a'])
        b = float(cif_block['_cell_length_b'])
        c = float(cif_block['_cell_length_c'])

        # 将分数坐标转换为笛卡尔坐标
        cartesian_coords = np.array([
            [a * x, b * y, c * z] for x, y, z in fractional_coords
        ])

        # 构建节点特征
        node_features = np.array([species_to_onehot[species] for species in atomic_species])

        # 生成边索引
        edge_index = []
        num_atoms = len(cartesian_coords)
        threshold = 3.0  # 距离阈值（例如 3.0 Å）

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = np.linalg.norm(cartesian_coords[i] - cartesian_coords[j])
                if dist < threshold:
                    edge_index.append([i, j])

        # 将数据转换为 PyTorch 格式
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 返回PyTorch Geometric的数据对象
        return Data(x=node_features, edge_index=edge_index)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


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
