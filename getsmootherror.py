import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import os
import asyncio


# 读取原始文件
def read_and_clean_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取数据部分
    start_data = False
    data_records = []

    for line in lines:
        if 'isotherm_data' in line:
            start_data = True
        elif start_data and 'pressure' in line:
            pressure = line.split(',')[1].strip().strip('"')
        elif start_data and 'adsorption' in line:
            adsorption = line.split(',')[1].strip().strip('"')
            data_records.append([pressure, adsorption])

    # 转换为DataFrame
    df = pd.DataFrame(data_records, columns=['Pressure (bar)', 'Adsorption (mmol/g)'])
    df['Pressure (bar)'] = pd.to_numeric(df['Pressure (bar)'], errors='coerce')
    df['Adsorption (mmol/g)'] = pd.to_numeric(df['Adsorption (mmol/g)'], errors='coerce')
    return df.dropna()


# 扩展数据并计算误差指标
def augment_data_and_calculate_metrics(df, new_points=1000, smoothing_factor=0.0011):
    # 移除重复和空值数据，确保数据合理性
    df = df.drop_duplicates()
    df = df.dropna()
    df = df[df['Pressure (bar)'] > 0]  # 过滤不合理的压力值

    # 按压力值排序，确保数据是递增的
    df = df.sort_values('Pressure (bar)')
    pressure = df['Pressure (bar)'].values
    adsorption = df['Adsorption (mmol/g)'].values

    # 样条插值拟合
    try:
        spline = UnivariateSpline(pressure, adsorption, s=smoothing_factor)  # 使用适当的平滑因子
        closest_smoothed_adsorption = spline(pressure)

        # 计算误差指标
        absolute_diff = np.abs(adsorption - closest_smoothed_adsorption)
        relative_error = absolute_diff / np.abs(adsorption)
        mse = np.mean(np.square(absolute_diff))

        # 返回误差指标
        return np.mean(absolute_diff), np.mean(relative_error), mse

    except Exception as e:
        print(f"插值时出错: {e}")
        return None, None, None


# 异步处理多个文件并计算误差指标
async def process_file(file_path, output_data):
    original_data = read_and_clean_data(file_path)
    absolute_diff, relative_error, mse = augment_data_and_calculate_metrics(original_data, new_points=1000)

    if absolute_diff is not None:
        print(f"{file_path} 的指标: 绝对差异={absolute_diff}, 相对误差={relative_error}, MSE={mse}")
        output_data.append([file_path, absolute_diff, relative_error, mse])
    return absolute_diff, relative_error, mse


# 保存误差数据到 CSV 文件
def save_error_data_to_csv(output_data, output_file_path):
    df = pd.DataFrame(output_data, columns=["File", "Absolute Difference", "Relative Error", "MSE"])
    df.to_csv(output_file_path, index=False)


# 主程序
async def main(directory_path, output_file_path):
    total_absolute_diff = 0
    total_relative_error = 0
    total_mse = 0
    file_count = 0

    output_data = []

    tasks = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"处理文件: {file_path}")
                tasks.append(process_file(file_path, output_data))

    results = await asyncio.gather(*tasks)

    for absolute_diff, relative_error, mse in results:
        total_absolute_diff += absolute_diff
        total_relative_error += relative_error
        total_mse += mse
        file_count += 1

    # 保存所有文件的误差数据到CSV
    save_error_data_to_csv(output_data, output_file_path)

    if file_count > 0:
        print(f"所有文件的平均绝对差异: {total_absolute_diff / file_count}")
        print(f"所有文件的平均相对误差: {total_relative_error / file_count}")
        print(f"所有文件的平均MSE: {total_mse / file_count}")
    else:
        print("没有有效的文件被处理")


if __name__ == "__main__":
    directory_path = './newdata/1/'  # 替换为实际的目录路径
    output_file_path = './newdata/error_metrics.csv'  # 保存误差数据的CSV文件路径
    asyncio.run(main(directory_path, output_file_path))
