import json
import numpy as np
import pandas as pd
import joblib
import websocket
from normalize_data import normalize_data
from interpolate_data import interpolate_data
import time  # 导入 time 模块
import matplotlib.pyplot as plt

# 加载 LDA 模型
lda = joblib.load('models/lda_model.pkl')

# 存储接收到的数据
data_buffer = []
is_recording = False
start_time = None  # 用于记录第一次接收到的数据的时间戳

# 定义 WebSocket 消息处理函数
def on_message(ws, message):
    global is_recording, data_buffer
    if message == "start":
        is_recording = True
        data_buffer = []  # 清空数据缓冲区
        start_time = time.time()  # 记录开始时间
        print("开始记录数据...")
    elif message == "end":
        is_recording = False
        print("结束记录数据...")

        # 将数据转换为 DataFrame
        df = pd.DataFrame(data_buffer)
        # 进行插值和归一化
        interpolated_data = interpolate_data(df,target_frames = 20)
        normalized_data = normalize_data(interpolated_data)

        # plt.figure(figsize=(12, 6))
        # # 绘制原始数据
        # plt.subplot(2, 1, 1)
        # plt.plot(df['time'], df['Ax'], label='original Ax', color='blue', marker='o', markersize=3)
        # plt.plot(df['time'], df['Ay'], label='original Ay', color='green', marker='o', markersize=3)
        # plt.plot(df['time'], df['Az'], label='original Az', color='red', marker='o', markersize=3)
        # plt.title(f'original')
        # plt.xlabel('time')
        # plt.ylabel('acceleration')
        # plt.legend()
        # plt.grid()
        # # 绘制处理后数据
        # plt.subplot(2, 1, 2)
        # plt.plot(normalized_data['time'], normalized_data['Ax'], label='interpolated Ax', color='blue', marker='o', markersize=3)
        # plt.plot(normalized_data['time'], normalized_data['Ay'], label='interpolated Ay', color='green', marker='o', markersize=3)
        # plt.plot(normalized_data['time'], normalized_data['Az'], label='interpolated Az', color='red', marker='o', markersize=3)
        # plt.title(f'interpolate')
        # plt.xlabel('time')
        # plt.ylabel('acceleration')
        # plt.legend()
        # plt.grid()

        # plt.tight_layout()
        # plt.show()  # 显示图形 
         
        # 将数据转换为特征数组
        features = [normalized_data[['Ax', 'Ay', 'Az', 'gx', 'gy', 'gz']].values.flatten()]
        # features = [normalized_data[['Ax', 'Ay', 'Az']].values.flatten()]
        # 使用 LDA 进行预测
        try:
            predicted_class = lda.predict(features)
            print(f"预测的类别: {predicted_class}")
            ws.send(predicted_class[0])  # 发送预测结果
        except Exception as e:
            print(f"预测时发生错误: {e}")
    else:
        if is_recording:
            # 解析 JSON 数据
            data = json.loads(message)
            # 收集数据
            ax = data['ax']
            ay = data['ay']
            az = data['az']
            gx = data['gx']
            gy = data['gy']
            gz = data['gz']

            # 计算相对时间戳
            current_time = time.time()
            # relative_time = current_time - start_time  # 计算相对时间

            # 将数据添加到缓冲区
            data_buffer.append({
                'id':0,
                'time': current_time,  # 使用相对时间戳
                'Ax': ax,
                'Ay': ay,
                'Az': az,
                'gx': gx,
                'gy': gy,
                'gz': gz
            })
            # print(len(data_buffer))

# 定义 WebSocket 连接打开时的处理函数
def on_open(ws):
    print("WebSocket 连接已打开")

# 定义 WebSocket 连接关闭时的处理函数
def on_close(ws):
    print("WebSocket 连接已关闭")

# WebSocket 连接地址
ws_url = "ws://192.168.31.113:8080"  # 替换为您的 ESP32 WebSocket 地址

# 创建 WebSocket 应用
ws = websocket.WebSocketApp(ws_url,
                            on_message=on_message,
                            on_open=on_open,
                            on_close=on_close)

# 运行 WebSocket
ws.run_forever()