import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor  # 导入GBDT回归器
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import time

data = pd.read_csv('C:/Users/Hasee/Desktop/沸石ml/mldata.csv', encoding='gbk',header=0)
data = data.dropna(how='all')
data = data[1:]
print(data.columns)

# 保留指定列
data = data[["total_adsorption(mmol/g)", "Temp", "Pressure(Bar)","Density(g/cm^3)","LCD","PLD","VSA(m^2/cm^3)","Vp(cm^3/g)","Void_fraction","E","Type"]]

# 确认是否有缺失值
print(data.isnull().sum())

# 删除包含缺失值的行
data = data.dropna()

# 确认是否删除成功
print(data.isnull().sum())

# 归一化
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
# 创建GBDT回归模型，你可以根据需要设置不同的超参数
start_time = time.time()
model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=10, random_state=42)

# 存储每次迭代的损失值
train_losses = []
test_losses = []
train_mae = []
test_mae = []
test_r2 = []
y_pred_train_cumulative = np.zeros(len(y_train))
y_pred_test_cumulative = np.zeros(len(y_test))

# 逐个迭代训练并预测
for i in range(1, 150):  # You can set the desired maximum iteration
    model.set_params(n_estimators=i)
    model.fit(X_train, y_train)

    # Training predictions and loss
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    train_losses.append(mse_train)
    y_pred_train_cumulative += y_pred_train

    # Test predictions and loss
    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    test_losses.append(mse_test)
    y_pred_test_cumulative += y_pred_test

    # Calculate MAE
    train_mae.append(mean_absolute_error(y_train, y_pred_train))
    test_mae.append(mean_absolute_error(y_test, y_pred_test))

    # Calculate R²
    test_r2.append(r2_score(y_test, y_pred_test))

    print(f"Iteration {i}, Train MSE: {mse_train}, Test MSE: {mse_test}")

# 最后一个模型的性能
final_mse_train = mean_squared_error(y_train, y_pred_train_cumulative / 200)  # 100为最大迭代次数
final_mse_test = mean_squared_error(y_test, y_pred_test_cumulative / 200)  # 100为最大迭代次数
print(f"Final Model Train MSE: {final_mse_train}")
print(f"Final Model Test MSE: {final_mse_test}")
end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算时间差
print(f'Training time: {elapsed_time} seconds')
# 绘制损失值的图表
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train MSE')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test MSE')
plt.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE')
plt.plot(range(1, len(test_mae) + 1), test_mae, label='Test MAE')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.title('Train and Test MSE/MAE')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(test_r2) + 1), test_r2, label='Test R²', color='orange')
plt.xlabel('Iteration')
plt.ylabel('R²')
plt.legend()
plt.title('Test R²')
plt.show()

# 训练模型
model.fit(X_train, y_train)

# 使用模型进行预测
y_pred = model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# 保存模型到文件
joblib.dump(model, 'gbdt_model.pkl')

# 对预测数据进行预测
# 加载模型
loaded_gbdt = joblib.load('gbdt_model.pkl')
prediction_get = loaded_gbdt.predict(X_test_tensor) #获取预测值
prediction_get = prediction_get.reshape(-1, 1)

X_test_tensor = scaler_x.inverse_transform(X_test_tensor) #反归一化
prediction = scaler_y.inverse_transform(prediction_get) #反归一化
# prediction_np = prediction.detach().numpy()
plt.figure()
pressure = X_test_tensor[:, 0]
reat_value = scaler_y.inverse_transform(y_test_tensor)[:, 0]
print(pressure.shape)
print(reat_value.shape)
plt.scatter(pressure, prediction, marker='o')
plt.scatter(pressure, reat_value, marker='x')
data = {
    'Pressure(Bar)': X_test_tensor[:, 0],
    'predictiont': prediction.flatten(),  # 将此列更改为模型的预测值
    'total_adsorption(mmol/g)': reat_value.flatten()  # 将此列更改为真实值
}

df = pd.DataFrame(data)
# 保存DataFrame为CSV文件
csv_filename = 'GBDTprediction_results3.csv'
df.to_csv(csv_filename, index=False)
print('====数据保存成功====')

# 添加标题和标签
plt.title('Predicted Results')
plt.xlabel('pressure')
plt.ylabel('Prediction')

plt.show()