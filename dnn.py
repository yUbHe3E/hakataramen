import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from torch import nn
import  time
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('C:/Users/Hasee/Desktop/沸石ml/mldata.csv', encoding='gbk',header=0)
data = data.dropna(how='all')
data = data[1:]
print(data.columns)

# 保留指定列
data = data[["total_adsorption(mmol/g)","Temp", "Pressure(Bar)","Density(g/cm^3)","LCD","PLD","VSA(m^2/cm^3)","Vp(cm^3/g)","Void_fraction","E","Type"]]

# 确认是否有缺失值
print(data.isnull().sum())

# 删除包含缺失值的行
data = data.dropna()

# 确认是否删除成功
print(data.isnull().sum())

# one_hot = OneHotEncoder()
# encoded = one_hot.fit_transform(data[["total_adsorption(mmol/g)"]])
#
# # 输出编码结果
# print(encoded.toarray())

#归一化
cols_to_convert = ["Temp", "Pressure(Bar)","Density(g/cm^3)","LCD","PLD","VSA(m^2/cm^3)","Vp(cm^3/g)","Void_fraction","E","Type"]
for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

#minmaxscaler to data
X = data[["Temp", "Pressure(Bar)","Density(g/cm^3)","LCD","PLD","VSA(m^2/cm^3)","Vp(cm^3/g)","Void_fraction","E","Type"]]
y = data[["total_adsorption(mmol/g)"]]
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaler = scaler_x.fit_transform(X)
y_scaler = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaler, y_scaler, train_size=0.8, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  # 将y_train改为y_test

# 创造数据集划分
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

assert X_train_tensor.size(0) == y_train_tensor.size(0), "Train sample size mismatch."
assert X_test_tensor.size(0) == y_test_tensor.size(0), "Test sample size mismatch."

print("X_test_tensor shape:", X_test_tensor.size())
print("y_test_tensor shape:", y_test_tensor.size())

#dnn模型
# 定义 DNN 模型
start_time = time.time()
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            #nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.ReLU()
        )
        '''
        self.layers = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )
        '''
    def forward(self, x):
        outputs = self.layers(x)
        return outputs

# 创建 DNN 模型实例
model = DNN()

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
def train(model, criterion, optimizer, train_loader, test_loader, num_epochs):
    train_losses = []
    test_losses = []
    train_mae = []  # 用于存储训练 MAE
    test_mae = []   # 用于存储测试 MAE
    test_r2 = []    # 用于存储测试 R²

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_absolute_error = 0.0  # 初始化 MAE

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 计算 MAE
            train_absolute_error += mean_absolute_error(labels.numpy(), outputs.detach().numpy())

            loss.backward()
            optimizer.step()

        train_losses.append(train_loss / len(train_loader))
        train_mae.append(train_absolute_error / len(train_loader))

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_absolute_error = 0.0  # 初始化 MAE
            all_predictions = []
            all_labels = []

            for i, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # 计算 MAE
                test_absolute_error += mean_absolute_error(labels.numpy(), outputs.detach().numpy())

                all_predictions.append(outputs)
                all_labels.append(labels)

            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # 计算 R²
            test_r2.append(r2_score(all_labels.numpy(), all_predictions.detach().numpy()))

        test_losses.append(test_loss / len(test_loader))
        test_mae.append(test_absolute_error / len(test_loader))

        # 打印损失、MAE 和 R²
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]}, Test MAE: {test_mae[-1]}, Test R²: {test_r2[-1]}')

        joblib.dump(model, "./skall.model")
        # 计算并打印出训练集和测试集的MSE
        with torch.no_grad():
            model.eval()
            train_mse = 0.0
            test_mse = 0.0

            for inputs, labels in train_loader:
                outputs = model(inputs)
                train_mse += criterion(outputs, labels).item()

            for inputs, labels in test_loader:
                outputs = model(inputs)
                test_mse += criterion(outputs, labels).item()

            train_mse /= len(train_loader)
            test_mse /= len(test_loader)
        print(f'Train MSE: {train_mse}, Test MSE: {test_mse}')

    return train_losses, test_losses, train_mae, test_mae, test_r2

# 调用 train 函数
train_losses, test_losses, train_mae, test_mae, test_r2 = train(model, criterion, optimizer, train_loader, test_loader, num_epochs=200)


end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算时间差

print(f'Training time: {elapsed_time} seconds')
# Plotting the training loss, test loss, training MAE, and test MAE
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train MSE')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test MSE')
plt.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE')
plt.plot(range(1, len(test_mae) + 1), test_mae, label='Test MAE')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()

# Plotting the test R²
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(test_r2) + 1), test_r2, label='Test R²', color='orange')
plt.xlabel('Iteration')
plt.ylabel('R²')
plt.legend()

plt.show()


model = joblib.load("./skall.model")

# 设置模型为评估模式
model.eval()

# 对预测数据进行预测

# sorted_tensor, indices = torch.sort(X_test_tensor, dim=0)
prediction_get = model(X_test_tensor).detach().numpy() #获取预测值
X_test_tensor = scaler_x.inverse_transform(X_test_tensor) #反归一化
prediction = scaler_y.inverse_transform(prediction_get) #反归一化
# prediction_np = prediction.detach().numpy()
plt.figure()
total_adsorption_mmol_g = X_test_tensor[:, 0]#np.sort( X_test_tensor[:, 0])
reat_value = scaler_y.inverse_transform(y_test_tensor)[:, 0]
print(total_adsorption_mmol_g.shape)
print(reat_value.shape)
plt.scatter(total_adsorption_mmol_g, prediction, marker='o')
plt.scatter(total_adsorption_mmol_g, reat_value, marker='x')
data = {
    'Pressure(Bar)': X_test_tensor[:, 0],
    'predictiont': prediction.flatten(),  # 将此列更改为模型的预测值
    'total_adsorption(mmol/g)': reat_value.flatten()  # 将此列更改为真实值
}


df = pd.DataFrame(data)

# # 保存DataFrame为CSV文件
# csv_filename = 'prediction_results2.csv'
# df.to_csv(csv_filename, index=False)
# print('====数据保存成功====')



# print(pressure)

# 添加标题和标签
plt.title('Predicted Results')
plt.xlabel('pressure')
plt.ylabel('Prediction')

# error=abs(reat_value-prediction)
# plt.figure()
# plt.plot(pressure, error)

# 显示图表
plt.show()