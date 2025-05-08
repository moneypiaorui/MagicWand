import pandas as pd
import numpy as np
import os  # 导入 os 模块
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # 导入 joblib
import json

from normalize_data import normalize_data
from interpolate_data import interpolate_data
from model import ActionClassifier,ActionClassifierCNN

TEST_SIZE = 0.25
TARGET_FRAMES = 10
num_epochs = 40

# 读取插值和归一化后的数据
def load_data(data_directory='data',target_frames = 100):
    labels = []
    features = []

    # 遍历 data 目录下的所有 CSV 文件
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            df = normalize_data(interpolate_data(pd.read_csv(file_path),target_frames = target_frames))  # 读取 CSV 文件、
            class_label = filename.split('.')[0]  # 使用文件名作为类别（去掉扩展名）

            # 根据 ID 分组并拼接特征
            for id_value, group in df.groupby('id'):
                # 将所有特征拼接在一起
                sample = group[['Ax', 'Ay', 'Az', 'gx', 'gy', 'gz']].values
                # sample = group[['Ax', 'Ay', 'Az']].values.flatten()
                features.append(sample)
                labels.append(class_label)
    # 创建 OneHotEncoder 对象并转换标签
    
    y_one_hot = one_hot_encoder.fit_transform(np.array(labels).reshape(-1, 1))

    return np.array(features), np.array(y_one_hot)

# 数据预处理
def preprocess_data(X, y,test_size=0.20):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=42)#random_state=42
    # X_train = X
    # y_train = y

    # # 标准化特征
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# 保存模型权重为 JSON 格式
def save_model_weights(model, filename):
    model_weights = {}
    for name, param in model.named_parameters():
        model_weights[name] = param.detach().numpy().tolist()  # 转换为列表以便于 JSON 序列化
    with open(filename, 'w') as f:
        json.dump(model_weights, f)

# 保存 OneHotEncoder 的类别信息
def save_one_hot_encoder(encoder, filename):
    # 将类别信息转换为列表
    categories = [cat.tolist() for cat in encoder.categories_]
    with open(filename, 'w') as f:
        json.dump(categories, f)

def save_model_config(model,filename):
    model_config = {}
    model_config['TARGET_FRAMES'] = TARGET_FRAMES
    for name, param in model.named_parameters():
        model_config[name] = len(param.detach().numpy().tolist())  # 转换为列表以便于 JSON 序列化
    with open(filename, 'w') as f:
        json.dump(model_config, f)


# 主程序
if __name__ == "__main__":
    # 加载数据
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    X, y = load_data(target_frames = TARGET_FRAMES)
    # for wei in X[0]:
    #     print(wei,end = ',')

    # 预处理数据
    X_train, X_test, y_train, y_test = preprocess_data(X,y,test_size=TEST_SIZE)

    # 保存 OneHotEncoder
    joblib.dump(one_hot_encoder, 'models/one_hot_encoder.pkl')  # 保存 OneHotEncoder

    # 转换为 PyTorch 张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test)

    print(f"输入 X 张量的维度: {X_train_tensor.shape[1]}， 输入label张量的维度: {y_train_tensor.shape[1]}")
    print(f"训练数量：{X_train_tensor.shape[0]}， 测试数量: {X_test_tensor.shape[0]}")
    # 加载数据
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

    # 构建模型
    model = ActionClassifier(X_train_tensor.shape,y_train_tensor.shape[1])

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 保存训练损失和测试准确率
    loss_values = []
    accuracy_values = []

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            
            # 测试模型
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                # print(test_outputs[0])
                _, predicted = torch.max(test_outputs.data, 1)
                _, y_correct = torch.max(y_test_tensor.data, 1)
                total = y_test_tensor.size(0)
                correct = (predicted == y_correct).sum().item() 
            # 保存测试准确率
            accuracy = 100 * correct / total
            accuracy_values.append(accuracy)
             # 保存当前损失
            loss_values.append(loss.item())

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, 测试准确率: {accuracy:.2f}%")

    # 绘制损失和准确率的曲线
    epochs = range(1, num_epochs + 1)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制训练损失
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss_values, label='Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个纵轴
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(epochs, accuracy_values, label='Test Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题和图例
    fig.suptitle('Training Loss and Test Accuracy Over Epochs')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # loss和accuracy关于frame-epoch的曲面图

    # frames = range(2, 20)
    # num_epochs = 40  # 假设每次运行的最大 epoch 数
    # loss_matrix = np.zeros((len(frames), num_epochs))  # 保存 loss 值
    # accuracy_matrix = np.zeros((len(frames), num_epochs))  # 保存 accuracy 值

    # for i, frame in enumerate(frames):
    #     X, y = load_data(target_frames=frame)
    #     X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=TEST_SIZE)

    #     # 转换为 PyTorch 张量
    #     X_train_tensor = torch.FloatTensor(X_train)
    #     X_test_tensor = torch.FloatTensor(X_test)
    #     y_train_tensor = torch.FloatTensor(y_train)
    #     y_test_tensor = torch.FloatTensor(y_test)

    #     train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    #     # 构建模型
    #     model = ActionClassifier(X_train_tensor.shape, y_train_tensor.shape[1])

    #     # 定义损失函数和优化器
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)

    #     for epoch in range(num_epochs):
    #         model.train()
    #         for features, labels in train_loader:
    #             optimizer.zero_grad()
    #             outputs = model(features)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()

    #         # 测试模型
    #         model.eval()
    #         with torch.no_grad():
    #             test_outputs = model(X_test_tensor)
    #             _, predicted = torch.max(test_outputs.data, 1)
    #             _, y_correct = torch.max(y_test_tensor.data, 1)
    #             total = y_test_tensor.size(0)
    #             correct = (predicted == y_correct).sum().item()
    #             accuracy = 100 * correct / total

    #         # 保存到矩阵中
    #         loss_matrix[i, epoch] = loss.item()
    #         accuracy_matrix[i, epoch] = accuracy

    # # 创建 frame 和 epoch 的网格
    # frame_grid, epoch_grid = np.meshgrid(frames, range(1, num_epochs + 1), indexing='ij')

    # # 绘制 Loss 曲面图
    # fig = plt.figure(figsize=(14, 6))

    # # Loss 曲面图
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.plot_surface(frame_grid, epoch_grid, loss_matrix, cmap='viridis', edgecolor='k')
    # ax1.set_title('Loss Surface')
    # ax1.set_xlabel('Target Frame')
    # ax1.set_ylabel('Epoch')
    # ax1.set_zlabel('Loss')

    # # Accuracy 曲面图
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.plot_surface(frame_grid, epoch_grid, accuracy_matrix, cmap='plasma', edgecolor='k')
    # ax2.set_title('Accuracy Surface')
    # ax2.set_xlabel('Target Frame')
    # ax2.set_ylabel('Epoch')
    # ax2.set_zlabel('Accuracy (%)')

    # plt.tight_layout()
    # plt.show()

        

    # 保存模型
    torch.save(model.state_dict(), 'models/action_classifier.pth')  # 保存模型的状态字典

    # 保存模型权重
    save_model_weights(model, 'models/model_weights.json')  # 保存模型权重为 JSON 文件

    # 保存 OneHotEncoder 的类别信息
    save_one_hot_encoder(one_hot_encoder, 'models/one_hot_encoder.json')  # 保存 OneHotEncoder 为 JSON 文件

    # save_model_config(model, 'models/model_config.json')
