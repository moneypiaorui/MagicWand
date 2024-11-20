import pandas as pd
import numpy as np
import os  # 导入 os 模块
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import joblib  # 导入 joblib
import matplotlib.pyplot as plt  # 导入 matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from normalize_data import normalize_data
from interpolate_data import interpolate_data

#可调整参数 preprocess_date的test_size ；load_data的target_dim

# name2label = {}
# label2name = []

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
                sample = group[['Ax', 'Ay', 'Az', 'gx', 'gy', 'gz']].values.flatten()
                # sample = group[['Ax', 'Ay', 'Az']].values.flatten()
                features.append(sample)
                # if class_label not in name2label:
                #     label_index = len(name2label)  # 获取当前标签的索引
                #     name2label[class_label] = label_index  # 将类名映射到整数标签
                #     label2name.append(class_label)  # 将类名添加到 label2name 列表
                # labels.append(name2label[class_label])  # 使用文件名作为标签
                labels.append(class_label)

    return np.array(features), np.array(labels)

# 数据预处理
def preprocess_data(X, y,test_size=0.20):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)#random_state=42
    # X_train = X
    # y_train = y

    # # 标准化特征
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# 主程序
if __name__ == "__main__":
    for dim in range(20,21,10):
        lda_acaccuracy = []
        knn_acaccuracy = []
        for i in range(2):
            # 加载数据
            X, y = load_data(target_frames = dim)
            
            # print(f"feature数量: {X.shape[0]},維度:{X.shape[1]}, label数量: {len(y)}")

            # # 打印每个类别的样本数量
            # print("每个类别的样本数量：")
            # print(pd.Series(y).value_counts())

            # 预处理数据
            X_train, X_test, y_train, y_test = preprocess_data(X, y,test_size=0.40)

            # 检查训练集和测试集的样本数量
            # print(f"训练集样本数量: {len(X_train)}, 测试集样本数量: {len(X_test)}")
            # print(f"训练集类别数量: {len(np.unique(y_train))}")

            # # 定义 LDA 的参数范围
            # lda_param_grid = {
            #     'n_components': range(2,len(np.unique(y_train))),
            # }

            # for i in lda_param_grid['n_components']:
            #     lda = LinearDiscriminantAnalysis(n_components = i)
            #     lda.fit(X_train, y_train)
            #      # 计算LDA准确率
            #     y_pred = lda.predict(X_test)
            #     accuracy = accuracy_score(y_test, y_pred)
            #     print(f"LDA n_components = {i}准确率: {accuracy:.2f}")

            # 创建 LDA 分类器
            lda = LinearDiscriminantAnalysis() 

            # 训练 LDA 模型
            lda.fit(X_train, y_train)

            # 保存 LDA 模型
            joblib.dump(lda, 'models/lda_model.pkl')  
            # print("LDA 模型已保存为 lda_model.pkl")

            # # 使用网格搜索进行超参数调优
            # lda_grid_search = GridSearchCV(lda , lda_param_grid, cv=5, scoring='accuracy')
            # lda_grid_search.fit(X_train, y_train)

            # # 输出最佳参数和最佳准确率
            # print(f"最佳参数: {lda_grid_search.best_params_}")
            # print(f"最佳准确率: {lda_grid_search.best_score_:.2f}")

            # # 使用最佳参数训练 KNN 模型
            # best_lda = lda_grid_search.best_estimator_
            # best_lda.fit(X_train, y_train)

            

            # 计算LDA准确率
            y_pred = lda.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # print(f"LDA 分类器测试准确率: {accuracy:.2f}")
            lda_acaccuracy.append(accuracy)

            # 使用 LDA 进行降维
            X_lda_train = lda.transform(X_train)  # 对训练集进行降维
            X_lda_test = lda.transform(X_test)    # 对测试集进行降维

            # # 定义 KNN 的参数范围
            # knn_param_grid = {
            #     'n_neighbors': [3, 5, 7, 9],
            #     'weights': ['uniform', 'distance']
            # }

            # 创建 KNN 分类器
            knn = KNeighborsClassifier(n_neighbors = 5,weights = 'uniform')
            knn.fit(X_lda_train, y_train)

            # # 使用网格搜索进行超参数调优
            # grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='accuracy')
            # grid_search.fit(X_lda_train, y_train)

            # # 输出最佳参数和最佳准确率
            # print(f"最佳参数: {grid_search.best_params_}")
            # print(f"最佳准确率: {grid_search.best_score_:.2f}")

            # # 使用最佳参数训练 KNN 模型
            # best_knn = grid_search.best_estimator_
            # best_knn.fit(X_lda_train, y_train)

            # 使用 KNN 进行预测
            y_pred = knn.predict(X_lda_test)
            accuracy = accuracy_score(y_test, y_pred)
            # print(f"KNN 分类器测试准确率: {accuracy:.2f}")
            knn_acaccuracy.append(accuracy)

            # 保存KNN模型
            joblib.dump(knn, 'models/knn_model.pkl')  
            # print("KNN 模型已保存为 knn_model.pkl")

        print(f"维度: {dim}")
        print(f"LDA 分类器平均准确率: {np.mean(lda_acaccuracy):.2f}")
        print(f"KNN 分类器平均准确率: {np.mean(knn_acaccuracy):.2f}")
    





        # 进行 3D 可视化
        lda = LinearDiscriminantAnalysis(n_components=3)  # 设置为 3 维
        lda.fit(X_train, y_train)
        X_lda = lda.transform(X)  # 将数据投影到 LDA 空间
        # 创建 3D 图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # 绘制每个类别的点
        for label in np.unique(y):
            ax.scatter(X_lda[y == label, 0], X_lda[y == label, 1], X_lda[y == label, 2], label=f'Class {label}')
        ax.set_title('LDA 3D Visualization')
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
        ax.legend()
        plt.show()

        lda = LinearDiscriminantAnalysis(n_components=2)  # 设置为 2 维
        lda.fit(X_train, y_train)
        X_lda = lda.transform(X)  # 将数据投影到 LDA 空间
        # 创建 2D 图形
        fig, ax = plt.subplots(figsize=(10, 8))
        # 绘制每个类别的点
        for label in np.unique(y):
            ax.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=f'Class {label}')
        ax.set_title('LDA 2D Visualization')
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.legend()
        plt.show()