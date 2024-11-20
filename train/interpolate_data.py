import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 线性插值
def interpolate_data(data, target_frames=100):
    interpolated_data = []

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

        # 仅在数据点数量足够时进行插值
        if len(time) > 1:  # 确保有足够的数据点进行插值
            target_time = np.linspace(time.min(), time.max(), target_frames)
            # 进行插值
            Ax_interp = np.interp(target_time, time, Ax)
            Ay_interp = np.interp(target_time, time, Ay)
            Az_interp = np.interp(target_time, time, Az)
            gx_interp = np.interp(target_time, time, gx)
            gy_interp = np.interp(target_time, time, gy)
            gz_interp = np.interp(target_time, time, gz)

            # 将插值结果添加到列表
            for t, ax, ay, az, gxi, gyi, gzi in zip(target_time, Ax_interp, Ay_interp, Az_interp, gx_interp, gy_interp, gz_interp):
                interpolated_data.append([id_value, t, ax, ay, az, gxi, gyi, gzi])

    # 转换为 DataFrame
    interpolated_df = pd.DataFrame(interpolated_data, columns=['id', 'time', 'Ax', 'Ay', 'Az', 'gx', 'gy', 'gz'])
    return interpolated_df

# 主程序
if __name__ == "__main__":
    for i in range(1,2):
        # 加载数据
        data = pd.read_csv(f"data/left_right.csv")

        # 进行插值
        interpolated_data = interpolate_data(data, target_frames=10)

        # 保存插值后的数据
        interpolated_data.to_csv(f'data/interpolated_left_right.csv', index=False)  
        print(f"插值后的数据已保存为 interpolated_left_right.csv")

        # 绘图对比
        for id_value in data['id'].unique():
            original_group = data[data['id'] == id_value]
            interpolated_group = interpolated_data[interpolated_data['id'] == id_value]

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
            plt.plot(interpolated_group['time'], interpolated_group['Ax'], label='interpolated Ax', color='blue', marker='o', markersize=3)
            plt.plot(interpolated_group['time'], interpolated_group['Ay'], label='interpolated Ay', color='green', marker='o', markersize=3)
            plt.plot(interpolated_group['time'], interpolated_group['Az'], label='interpolated Az', color='red', marker='o', markersize=3)
            plt.title(f'interpolate (ID: {id_value})')
            plt.xlabel('time')
            plt.ylabel('acceleration')
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()  # 显示图形 