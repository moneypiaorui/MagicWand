import torch
import torch.nn as nn
import os

# 构建神经网络模型
class ActionClassifier(nn.Module):
    def __init__(self,input_shape,output_classes):
        super(ActionClassifier, self).__init__()

        self.fc1 = nn.Linear(input_shape[1]*input_shape[2], 32)  # 输入特征数
        # self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_classes)  # 输出层节点数为CSV文件数量

    def forward(self, x):
        x = torch.flatten(x,start_dim=-2)
        x = self.fc1(x)
        x = torch.relu(x)
        # x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.fc3(x)
        return x
    
# CNN模型
class ActionClassifierCNN(nn.Module):
    def __init__(self,input_shape,output_classes):
        super(ActionClassifierCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=input_shape[2], out_channels=30, kernel_size=3, stride=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=30, out_channels=15, kernel_size=3, stride=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(15 * ((input_shape[1]-1)//9+1), output_classes)  
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.flatten(x,start_dim=-2)# 倒数第二个维度开始，批量和单个数据都可以运行
        x = self.fc1(x)
        x = self.dropout(x)
        return x  # 分类问题使用 softmax 输出