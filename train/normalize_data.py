import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 归一化数据
def normalize_data(data):
    normalized_data = []

    # 根据 ID 分组
    grouped = data.groupby('id')

    for id_value, group in grouped:
        # 获取时间和特征
        time = group['time'].values
        Ax = group['Ax'].values
        Ay = group['Ay'].values
        Az = group['Az'].values
        gx = group['gx'].values
        gy = group['gy'].values
        gz = group['gz'].values

        # 归一化时间
        time_normalized = np.linspace(0, 1, num=len(time))  # 归一化到 [0, 1]

        # 将数据拼接在一起
        for t in time_normalized:
            idx = np.searchsorted(time_normalized, t)
            if idx < len(time):
                normalized_data.append([id_value, t, Ax[idx], Ay[idx], Az[idx], gx[idx], gy[idx], gz[idx]])

        # 将每个维度归一化到 [-1, 1] 范围
        # def scale_to_range(data_array):
        #     min_val = np.min(data_array)
        #     max_val = np.max(data_array)
        #     # 避免除以 0
        #     if max_val - min_val == 0:
        #         return data_array  # 若最大最小相等，直接返回原数据
        #     return 2 * (data_array - min_val) / (max_val - min_val) - 1

        # Ax_normalized = scale_to_range(Ax)
        # Ay_normalized = scale_to_range(Ay)
        # Az_normalized = scale_to_range(Az)
        # gx_normalized = scale_to_range(gx)
        # gy_normalized = scale_to_range(gy)
        # gz_normalized = scale_to_range(gz)

        # # 将数据拼接在一起
        # for idx, t in enumerate(time_normalized):
        #     normalized_data.append([
        #         id_value, t,
        #         Ax_normalized[idx], Ay_normalized[idx], Az_normalized[idx],
        #         gx_normalized[idx], gy_normalized[idx], gz_normalized[idx]
        #     ])

    # 转换为 DataFrame
    normalized_df = pd.DataFrame(normalized_data, columns=['id', 'time', 'Ax', 'Ay', 'Az', 'gx', 'gy', 'gz'])
    return normalized_df

# 主程序
if __name__ == "__main__":
    for i in range(1, 2):
        # 加载数据
        data = pd.read_csv(f"data/interpolated_left_right.csv")

        # 归一化数据
        normalized_data = normalize_data(data)

        # 保存归一化后的数据
        normalized_data.to_csv(f'data/normalized_left_right.csv', index=False)
        print(f"归一化后的数据已保存为 normalized_left_right.csv") 

        # 绘图对比
        for id_value in data['id'].unique():
            original_group = data[data['id'] == id_value]
            normalized_group = normalized_data[normalized_data['id'] == id_value]

            plt.figure(figsize=(12, 6))

            # 绘制原始数据
            plt.subplot(2, 1, 1)
            plt.plot(original_group['time'], original_group['Ax'], label='original Ax', color='blue', marker='o', markersize=3)
            plt.plot(original_group['time'], original_group['Ay'], label='original Ay', color='green', marker='o', markersize=3)
            plt.plot(original_group['time'], original_group['Az'], label='original Az', color='red', marker='o', markersize=3)
            plt.title(f'original (ID: {id_value})')
            plt.xlabel('time')
            plt.ylabel('acceleration')
            plt.legend()
            plt.grid()

            # 绘制插值数据
            plt.subplot(2, 1, 2)
            plt.plot(normalized_group['time'], normalized_group['Ax'], label='downsampled Ax', color='blue', marker='o', markersize=3)
            plt.plot(normalized_group['time'], normalized_group['Ay'], label='downsampled Ay', color='green', marker='o', markersize=3)
            plt.plot(normalized_group['time'], normalized_group['Az'], label='downsampled Az', color='red', marker='o', markersize=3)
            plt.title(f' normalized (ID: {id_value})')
            plt.xlabel('time')
            plt.ylabel('acceleration')
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()  # 显示图形 