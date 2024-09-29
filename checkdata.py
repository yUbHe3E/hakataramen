import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# 读取数据
data_frame = pd.read_excel('database-co2.xlsx')
Temp,Temp_lambda = stats.boxcox(data_frame['Temp'] + 1e-20)
Pressure, pressure_lambda = stats.boxcox(data_frame['Pressure(Bar)'] + 1e-20)
Asor, adsorption_lambda = stats.boxcox(data_frame['total_adsorption(mmol/g)'] + 1e-20)
# Type, Type_lambda = stats.boxcox(data_frame['Type'])
print(Temp_lambda,pressure_lambda,adsorption_lambda,)
# 绘制温度的直方图
plt.hist(Temp, bins=30, edgecolor='k')
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()

# 绘制压力的直方图
plt.hist(Pressure, bins=30, edgecolor='k')
plt.title('Pressure Distribution')
plt.xlabel('Pressure (Bar)')
plt.ylabel('Frequency')
plt.show()

# 绘制吸附量的直方图
plt.hist(Asor, bins=30, edgecolor='k')
plt.title('Adsorption Distribution')
plt.xlabel('Adsorption (mmol/g)')
plt.ylabel('Frequency')
plt.show()

# plt.hist(Type, bins=30, edgecolor='k')
# plt.title('Type')
# plt.xlabel('Type')
# plt.ylabel('Frequency')
# plt.show()

# 绘制温度的直方图
plt.hist(data_frame['Temp'], bins=30, edgecolor='k')
plt.title('Temperature Distribution1')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()

# 绘制压力的直方图
plt.hist(data_frame['Pressure(Bar)'], bins=30, edgecolor='k')
plt.title('Pressure Distribution1')
plt.xlabel('Pressure (Bar)')
plt.ylabel('Frequency')
plt.show()

# 绘制吸附量的直方图
plt.hist(data_frame['total_adsorption(mmol/g)'], bins=30, edgecolor='k')
plt.title('Adsorption Distribution1')
plt.xlabel('Adsorption (mmol/g)')
plt.ylabel('Frequency')
plt.show()
#
# plt.hist(data_frame['Type'], bins=30, edgecolor='k')
# plt.title('Type')
# plt.xlabel('Type')
# plt.ylabel('Frequency')
# plt.show()
# import seaborn as sns
#
# # 温度的箱线图
# sns.boxplot(y=data_frame['Temp'])
# plt.title('Temperature Boxplot')
# plt.show()
#
# # 压力的箱线图
# sns.boxplot(y=data_frame['Pressure(Bar)'])
# plt.title('Pressure Boxplot')
# plt.show()
#
# # 吸附量的箱线图
# sns.boxplot(y=data_frame['total_adsorption(mmol/g)'])
# plt.title('Adsorption Boxplot')
# plt.show()
# # 温度的KDE图
# sns.kdeplot(data_frame['Temp'], shade=True)
# plt.title('Temperature Density Plot')
# plt.show()
#
# # 压力的KDE图
# sns.kdeplot(data_frame['Pressure(Bar)'], shade=True)
# plt.title('Pressure Density Plot')
# plt.show()
#
# # 吸附量的KDE图
# sns.kdeplot(data_frame['total_adsorption(mmol/g)'], shade=True)
# plt.title('Adsorption Density Plot')
# plt.show()
# 温度的统计量
# print('Temperature Statistics:')
# print(Temp.tolist().describe())
# print('Skewness:', Temp.tolist().skew())
# print('Kurtosis:', Temp.tolist().kurt())
#
# # 压力的统计量
# print('\nPressure Statistics:')
# print(Temp.tolist().describe())
# print('Skewness:', Pressure.tolist().skew())
# print('Kurtosis:', Pressure.tolist().kurt())
#
# # 吸附量的统计量
# print('\nAdsorption Statistics:')
# print(Asor.describe())
# print('Skewness:', Asor.tolist().skew())
# print('Kurtosis:', Asor.tolist().kurt())
