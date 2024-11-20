import json
import numpy as np
import pandas as pd
import joblib
import websocket
import time  # 导入 time 模块
import matplotlib.pyplot as plt
import torch

from normalize_data import normalize_data
from interpolate_data import interpolate_data
from model import ActionClassifier,ActionClassifierCNN

# 加载模型
def load_model(model_path='models/action_classifier.pth'):
    model = ActionClassifier([0,20,6],12)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model

# 加载 OneHotEncoder
one_hot_encoder = joblib.load('models/one_hot_encoder.pkl')  # 加载 OneHotEncoder

# 进行预测
def predict(model, input_data):
    with torch.no_grad():
        input_tensor = torch.FloatTensor([input_data])
        try:
            [outputs] = model(input_tensor)
            print([f"{x * 100:.2f}%" for x in torch.softmax(outputs, dim=0)])
            one_hot_output = torch.zeros_like(outputs)
            one_hot_output[torch.argmax(outputs)] = 1
            predicted_labels = one_hot_encoder.inverse_transform(one_hot_output.unsqueeze(0))
            return predicted_labels.item()

        except Exception as e:
            print("预测时出错:", e)
            return None

# 加载模型
model = load_model()
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
        interpolated_data = interpolate_data(df, target_frames=20)
        normalized_data = normalize_data(interpolated_data)

        # plt.plot(normalized_data['time'], normalized_data['Ax'], label='interpolated Ax', color='blue', marker='o', markersize=3)
        # plt.plot(normalized_data['time'], normalized_data['Ay'], label='interpolated Ay', color='green', marker='o', markersize=3)
        # plt.plot(normalized_data['time'], normalized_data['Az'], label='interpolated Az', color='red', marker='o', markersize=3)
        # plt.title(f'normalize ')
        # plt.xlabel('time')
        # plt.ylabel('acceleration')
        # plt.legend()
        # plt.grid()
        # plt.show() 
         
        # 将数据转换为特征数组
        input_data = normalized_data[['Ax', 'Ay', 'Az', 'gx', 'gy', 'gz']].values
        # 进行预测
        predictions = predict(model, input_data)
        ws.send(predictions)  # 发送预测结果
        # 输出预测结果
        print(f"预测的类别: {predictions}") 

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
                'id': 0,
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
ws_url = "ws://192.168.31.12:8080"  # 替换为您的 ESP32 WebSocket 地址

# 创建 WebSocket 应用
ws = websocket.WebSocketApp(ws_url,
                            on_message=on_message,
                            on_open=on_open,
                            on_close=on_close)

# 运行 WebSocket
ws.run_forever()