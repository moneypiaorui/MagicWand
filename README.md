# 基于魔杖控制的智能家居系统
本项目是基于ESP32和HomeAssistant的魔杖智能家具控制系统中的魔杖设计部分；通过神经网络将采集到的MPU6050原始数据进行12分类，然后通过MQTT集成通知HomeAssistant，在HA中配置自动化逻辑
## 文件结构
- /main：Arduino的源代码
  - /build：编译后的二进制文件
  - /data：网页文件和神经网络配置文件
- /train：训练的python脚本
  - /model.py：神经网络的定义
  - /ANN.py：模型训练
  - /predictor_ANN.py：使用ws连接ESP32进行预测的相关测试
  - /data：训练用的的csv数据
- /other：动作设计图以及七八糟的东西
- /libraries：Arduino需要的库
## 相关资料
参考项目：https://oshwhub.com/lyg0927/cyberwand-stm32-convolutional-ne

技术文档：https://chainpray.top/index.php/2024/11/09/%e9%ad%94%e6%9d%96%e6%8a%80%e6%9c%af%e6%96%87%e6%a1%a3/

立创开源：https://oshwhub.com/piaoray/magicwand