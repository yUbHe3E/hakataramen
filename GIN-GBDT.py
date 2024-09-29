import torch
from torch_geometric.loader import DataLoader
from GIN import FullModel  # 假设你将之前的 GIN 模型保存为 GINModel.py
from GINdataset import AdsorptionDataset  # 数据集类与你的 ALIGNN 模型中一致
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 生成 GIN 嵌入特征的函数
def generate_gin_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            temp_pressure = batch_data.temp_pressure.to(device)
            graph_embedding = model(batch_data, temp_pressure)
            embeddings.append(graph_embedding.cpu().numpy())
            labels.append(batch_data.y.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels

# 定义训练 GBDT 的函数
def train_gbdt(gbdt_model, train_embeddings, train_labels, val_embeddings, val_labels):
    gbdt_model.fit(train_embeddings, train_labels)
    val_predictions = gbdt_model.predict(val_embeddings)
    val_loss = mean_squared_error(val_labels, val_predictions)
    return val_loss, val_predictions

# 主训练循环
def main():
    # 归一化参数
    mean_temp_pressure = torch.tensor([302.71597633, 0.7167231, 2.65004227])
    std_temp_pressure = torch.tensor([19.18166408, 1.37877475, 1.62948257])
    mean_adsorption = torch.tensor(1.2285861588841926)
    std_adsorption = torch.tensor(1.2555190110695644)

    # 加载数据集
    csv_file = 'database.xlsx'
    cif_directory = './cif_file/'
    dataset = AdsorptionDataset(
        csv_file=csv_file,
        cif_directory=cif_directory,
        mean_temp_pressure=mean_temp_pressure,
        std_temp_pressure=std_temp_pressure,
        mean_adsorption=mean_adsorption,
        std_adsorption=std_adsorption
    )

    # 使用 DataLoader 加载数据
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=43, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=79, shuffle=True)

    # 设备选择 (GPU 或 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化 GIN 模型
    model = FullModel(
        gin_node_in_dim=13,  # 假设每个节点的特征维度为 13
        gin_node_hidden_dim=32,
        gin_num_layers=2,
        gin_output_dim=64,  # 嵌入维度
        temp_pressure_dim=3  # 温度和压力是 3 维向量
    ).to(device)

    # 生成 GIN 嵌入特征
    train_embeddings, train_labels = generate_gin_embeddings(model, train_loader, device)
    val_embeddings, val_labels = generate_gin_embeddings(model, val_loader, device)

    # 初始化 GBDT 模型
    gbdt_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=10, random_state=42)

    # 使用 GBDT 训练
    best_val_loss, val_predictions = train_gbdt(gbdt_model, train_embeddings, train_labels, val_embeddings, val_labels)

    print(f"Validation MSE after GBDT training: {best_val_loss}")

    # 保存为 CSV 文件
    data = {'predictions': val_predictions, 'real': val_labels}
    df = pd.DataFrame(data)
    df.to_csv("predictions_gbdt.csv", index=False)

if __name__ == '__main__':
    main()
