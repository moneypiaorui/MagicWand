
#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <PubSubClient.h>  // 添加 MQTT 库
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <ArduinoJson.h>
#include <ArduinoOTA.h>
#include <LittleFS.h>
#include <vector>
#include <algorithm>

String wifi_config_file = "/wifiConfig.txt";
String mqtt_config_file = "/mqttConfig.txt";

String ssid = "ChainPray";        // 你的Wi-Fi SSID
String password = "qdd20050629";  // 你的Wi-Fi密码

String mqttServer = "118.178.255.236";
String mqttPort = "1883";
String mqttUser = "admin";
String mqttPassword = "public";
const char* controlTopic = "magicWind/control";
const char* topic1 = "magicWind/command/1";
const char* topic2 = "magicWind/command/2";

std::vector<String> actions;

const int layer1 = 120;
const int layer2 = 32;
int layer3 = 16;
float fc1_weight[layer2 * layer1];
float fc1_bias[layer2];
float* fc3_weight;
float* fc3_bias;

// 创建 Adafruit_MPU6050 对象
Adafruit_MPU6050 mpu;
WebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(8080);
std::vector<int> clients;  // 存储连接的客户端编号
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);  // 创建 MQTT 客户端

#define SDA_PIN 4    // 自定义 SDA 引脚
#define SCL_PIN 5    // 自定义 SCL 引脚
#define TOUCH_PIN 0  // 触摸引脚（可以根据实际情况选择）
#define LED_PIN 6

bool isRecording = false;  // 是否正在记录数据

struct SensorData {
  float time;        // 时间
  float Ax, Ay, Az;  // 加速度
  float gx, gy, gz;  // 陀螺仪数据
                     // 构造函数初始化结构体成员
  SensorData(float time = 0.0f, float ax = 0.0f, float ay = 0.0f, float az = 0.0f,
             float gx = 0.0f, float gy = 0.0f, float gz = 0.0f)
    : time(time), Ax(ax), Ay(ay), Az(az), gx(gx), gy(gy), gz(gz) {}
};
std::vector<SensorData> actionRecord;
// 定义 ReLU 激活函数
float relu(float x) {
  return (x > 0) ? x : 0;
}
// Softmax 函数
void softmax(float* output, int length) {
  float sum = 0.0;
  // 计算所有输出值的指数和
  for (int i = 0; i < length; i++) {
    sum += exp(output[i]);
  }

  // 将每个输出转化为概率
  for (int i = 0; i < length; i++) {
    output[i] = exp(output[i]) / sum;
  }
}
// 线性插值函数
float linear_interpolation(float target_time, float time0, float time1, float value0, float value1) {
  return value0 + (value1 - value0) * (target_time - time0) / (time1 - time0);
}
// 全连接层操作
float fc(float* input, float* weights, float* bias, int input_size) {
  float output = 0;
  for (int i = 0; i < input_size; i++) {
    output += input[i] * weights[i];
  }
  return output + bias[0];
}
// 前向传播逻辑
void forward(float* input, float* output, float* fc1_weight, float* fc1_bias, float* fc3_weight, float* fc3_bias) {
  float hidden[layer2];

  // 第一层全连接 (fc1)
  for (int i = 0; i < layer2; i++) {
    hidden[i] = 0;
    for (int j = 0; j < layer1; j++) {
      hidden[i] += input[j] * fc1_weight[i * layer1 + j];
    }
    hidden[i] += fc1_bias[i];
    hidden[i] = relu(hidden[i]);  // ReLU 激活函数
  }

  // 第三层全连接 (fc3)
  for (int i = 0; i < layer3; i++) {
    output[i] = 0;
    for (int j = 0; j < layer2; j++) {
      output[i] += hidden[j] * fc3_weight[i * layer2 + j];
    }
    output[i] += fc3_bias[i];
  }
}
// 插值降维
std::vector<SensorData> interpolate_data(const std::vector<SensorData>& data, int target_frames = 20) {
  std::vector<SensorData> interpolated_data;

  // 根据时间排序数据
  std::vector<SensorData> sorted_data = data;
  std::sort(sorted_data.begin(), sorted_data.end(), [](const SensorData& a, const SensorData& b) {
    return a.time < b.time;
  });

  // 确保数据点数量大于1，否则无法插值
  if (sorted_data.size() > 1) {
    // 获取时间范围
    float min_time = sorted_data[0].time;
    float max_time = sorted_data[sorted_data.size() - 1].time;

    // 计算插值时间点
    std::vector<float> target_time;
    for (int i = 0; i < target_frames; ++i) {
      target_time.push_back(min_time + i * (max_time - min_time) / (target_frames - 1));
    }

    // 对每个目标时间进行插值
    for (float t : target_time) {
      // 找到对应的时间点区间进行线性插值
      for (size_t i = 0; i < sorted_data.size() - 1; ++i) {
        if (t >= sorted_data[i].time && t <= sorted_data[i + 1].time) {
          float Ax_interp = linear_interpolation(t, sorted_data[i].time, sorted_data[i + 1].time, sorted_data[i].Ax, sorted_data[i + 1].Ax);
          float Ay_interp = linear_interpolation(t, sorted_data[i].time, sorted_data[i + 1].time, sorted_data[i].Ay, sorted_data[i + 1].Ay);
          float Az_interp = linear_interpolation(t, sorted_data[i].time, sorted_data[i + 1].time, sorted_data[i].Az, sorted_data[i + 1].Az);
          float gx_interp = linear_interpolation(t, sorted_data[i].time, sorted_data[i + 1].time, sorted_data[i].gx, sorted_data[i + 1].gx);
          float gy_interp = linear_interpolation(t, sorted_data[i].time, sorted_data[i + 1].time, sorted_data[i].gy, sorted_data[i + 1].gy);
          float gz_interp = linear_interpolation(t, sorted_data[i].time, sorted_data[i + 1].time, sorted_data[i].gz, sorted_data[i + 1].gz);

          // 将插值结果保存到新结构体中
          SensorData new_data = { t, Ax_interp, Ay_interp, Az_interp, gx_interp, gy_interp, gz_interp };
          interpolated_data.push_back(new_data);
          break;  // 找到合适的时间区间后跳出
        }
      }
    }
  }

  return interpolated_data;
}

void setup() {
  Serial.begin(115200);
  delay(10);
  pinMode(TOUCH_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, 1);


  //启动闪存服务
  if (LittleFS.begin()) {
    Serial.println("LittleFS Started.");
  } else {
    Serial.println("LittleFS Failed to Started.");
  }

  String logfilename = "/setup.log";
  // 如果log文件已存在，删除旧文件
  if (LittleFS.exists(logfilename)) {
    LittleFS.remove(logfilename);
  }
  // 打开文件进行写入
  File logfile = LittleFS.open(logfilename, "w");

  getWIFIconfig();
  getMQTTconfig();
  loadOneHotEncoder("/one_hot_encoder.json");
  // 根据actions的数量设置输出层节点数
  layer3 = actions.size();
  fc3_weight = new float[layer3 * layer2];
  fc3_bias = new float[layer3];
  // 然后读取权重
  loadModelWeights("/model_weights.json");

  // 尝试连接 Wi-Fi
  logfile.println("ssid:" + ssid);
  logfile.println("password:" + password);
  Serial.println("ssid:" + ssid);
  Serial.println("password:" + password);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  logfile.print("Connecting to WiFi...");
  int connectTimes = 0;
  while (WiFi.status() != WL_CONNECTED && connectTimes < 5) {
    digitalWrite(LED_PIN, 0);
    delay(500);
    digitalWrite(LED_PIN, 1);
    delay(500);
    Serial.print(".");
    logfile.print(".");
    connectTimes++;
  }
  // 如果连接失败，则开启热点
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Not connected!");
    logfile.println("Not connected!");
    WiFi.softAP("ESP32_Hotspot");
    Serial.println("Hotspot created!");
    logfile.println("Hotspot created!");
  } else {
    Serial.println("Connected!");
    logfile.println("Connected!");
  }
  // 打印 IP 地址
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // 设置 WebSocket 回调
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);

  // 设置web服务器路由
  server.on("/WIFIconfig", handleWIFI);
  server.on("/MQTTconfig", handleMQTT);
  server.on("/restart", handleRestart);
  server.on("/upload", HTTP_POST, handleFileUpload, handleUploadForm);  // 处理文件上传请求
  server.on("/update", HTTP_POST, handleFirmwareUpload, handleUpdateBin);  // 处理 OTA 请求
  server.onNotFound(handleUserRequet);  //处理没有匹配的处理程序的url
  server.begin();
  Serial.println("HTTP server started");
  logfile.println("HTTP server started");

  // 初始化 I2C，引脚可以自定义
  Wire.begin(SDA_PIN, SCL_PIN);

  // 初始化 MPU6050
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    logfile.println("Failed to find MPU6050 chip");
    // while (1) { delay(100); }
  } else {
    Serial.println("MPU6050 found!");
    logfile.println("MPU6050 found!");
  }


  // MQTT 设置
  mqttClient.setServer(mqttServer.c_str(), mqttPort.toInt());
  mqttClient.setCallback(callback);

  // touchAttachInterrupt(TOUCH_PIN, touchCallback, 40);  // 设置触摸中断

  // 启动 OTA 功能
  ArduinoOTA.begin();

  digitalWrite(LED_PIN, 0);
  logfile.close();
}

unsigned long lastSendTime = 0;
const unsigned long sendInterval = 10;  // 发送间隔（毫秒）

void loop() {
  server.handleClient();
  ArduinoOTA.handle();
  webSocket.loop();
  if (!mqttClient.connected()) {
    digitalWrite(LED_PIN, 1);
    reconnectMQTT();
    digitalWrite(LED_PIN, 0);
  }
  mqttClient.loop();



  if (digitalRead(TOUCH_PIN) == 0) {
    if (!isRecording) {
      for (auto clientNum : clients) {
        webSocket.sendTXT(clientNum, "start");  // 发送开始标志给每个客户端
      }
      isRecording = true;
      digitalWrite(LED_PIN, 1);
      Serial.println("Touch on.");
    }
  } else {
    if (isRecording) {
      for (auto clientNum : clients) {
        webSocket.sendTXT(clientNum, "end");  // 发送结束标志给每个客户端
      }
      isRecording = false;
      digitalWrite(LED_PIN, 0);
      Serial.println("Touch off.");
      String predictResult = actionPredict();
      mqttClient.publish(controlTopic, predictResult.c_str());
      Serial.print("Predict action:");
      Serial.println(predictResult);
    }
  }

  unsigned long currentTime = millis();
  if (currentTime - lastSendTime >= sendInterval) {
    lastSendTime = currentTime;
    if (isRecording) {
      sendMPU6050Data();
    }
  }
}

String actionPredict() {
  std::vector<SensorData> actionInput = interpolate_data(actionRecord, layer1 / 6);
  float* input_data = new float[layer1];
  float output[layer3];
  // 将 actionRecord 中的数据展开到 input_data
  size_t idx = 0;
  for (const SensorData& record : actionInput) {
    input_data[idx++] = record.Ax;
    input_data[idx++] = record.Ay;
    input_data[idx++] = record.Az;
    input_data[idx++] = record.gx;
    input_data[idx++] = record.gy;
    input_data[idx++] = record.gz;
  }
  // 执行前向传播
  forward(input_data, output, fc1_weight, fc1_bias, fc3_weight, fc3_bias);
  // 执行 softmax 函数以获取预测概率
  softmax(output, layer3);
  // 计算最大概率值及其索引
  float maxProbability = -1.0;
  int predictIndex = -1;
  for (int i = 0; i < layer3; i++) {
    if (maxProbability < output[i]) {
      maxProbability = output[i];
      predictIndex = i;
    }
  }

  // Serial.print("original frames:");
  // Serial.println(actionRecord.size());
  // Serial.print("input frames:");
  // Serial.println(actionInput.size());
  // for (SensorData frame : actionRecord) {
  //   Serial.print(String(frame.Ax) + ",");
  // }
  // Serial.println();
  // for (SensorData frame : actionInput) {
  //   Serial.print(String(frame.Ax) + ",");
  // }
  // Serial.println();
  // Serial.print("input dim:");
  // Serial.println(idx);
  // for (int i = 0; i < idx; i++) {
  //   Serial.print(String(input_data[i]) + ",");
  // }
  // Serial.println();
  // for (int i = 0; i < layer3; i++) {
  //   Serial.print(output[i]);
  //   Serial.print(", ");
  // }
  // Serial.println();

  delete[] input_data;
  actionRecord.clear();
  // 如果最大概率大于 80%，返回预测动作；否则返回 "none"
  if (maxProbability > 0.8) {
    return actions[predictIndex];
  } else {
    return "none";
  }
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_DISCONNECTED) {
    Serial.printf("Client %u disconnected\n", num);
    clients.erase(std::remove(clients.begin(), clients.end(), num), clients.end());
  } else if (type == WStype_CONNECTED) {
    Serial.printf("Client %u connected\n", num);
    clients.push_back(num);
  } else if (type == WStype_TEXT) {
    // 接收到文本消息时，使用 MQTT 发布
    String message = String((char*)payload).substring(0, length);
    for (String action : actions) {
      if (message == action) {
        mqttClient.publish(controlTopic, action.c_str());
        break;
      }
    }

    Serial.printf("Received message: %s\n", message.c_str());
  }
}

void sendMPU6050Data() {

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  SensorData newActionData = { float(lastSendTime), a.acceleration.x, a.acceleration.y, a.acceleration.z, g.gyro.x, g.gyro.y, g.gyro.z };
  actionRecord.push_back(newActionData);

  // 构建 JSON 格式的数据
  String json = String("{\"ax\":") + a.acceleration.x + ",\"ay\":" + a.acceleration.y + ",\"az\":" + a.acceleration.z + ",\"gx\":" + g.gyro.x + ",\"gy\":" + g.gyro.y + ",\"gz\":" + g.gyro.z + "}";
  // 遍历所有客户端并发送消息
  for (auto clientNum : clients) {
    webSocket.sendTXT(clientNum, json);  // 发送消息给每个客户端
  }
}

void reconnectMQTT() {
  // 循环直到重新连接
  // while (!mqttClient.connected()) {
  Serial.print("Connecting to MQTT ");
  Serial.print(mqttServer.c_str());
  if (mqttClient.connect("magicWindClient", mqttUser.c_str(), mqttPassword.c_str())) {
    Serial.println(" connected");
    publishDeviceConfiguration();
    mqttClient.subscribe(topic1);
  } else {
    Serial.print(" failed, rc=");
    Serial.print(mqttClient.state());
    Serial.println(" try again in 0.2 seconds");
    delay(200);
  }
  // }
}

void callback(char* topic, byte* payload, unsigned int length) {
  // 处理订阅消息的回调函数
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  char* cmd = new char[length + 1];
  for (int i = 0; i < length; i++) {
    cmd[i] = (char)payload[i];
  }
  cmd[length] = '\0';
  Serial.println(cmd);
  //  client.publish("esp8266/callback", cmd);//这一句会让topic直接乱码
  if (strcmp(topic, topic1) == 0) {

  } else if (strcmp(topic, topic2) == 0) {
  }
  delete[] cmd;  // 释放动态分配的内存
  Serial.println();
}

void publishDeviceConfiguration() {
  // 通过MQTT发现，自动配置触发器(trigger)
  for (String action : actions) {
    Serial.print(action);
    Serial.print(",");
    StaticJsonDocument<512> doc;  // 在每次循环中创建新的 JSON 对象
    doc["automation_type"] = "trigger";
    doc["type"] = "action";
    doc["subtype"] = action;  // 设置 subtype
    doc["payload"] = action;  // 设置 payload
    doc["topic"] = controlTopic;

    JsonObject device = doc.createNestedObject("device");
    device["identifiers"][0] = "magic_wind";  // 使用小写和下划线
    device["name"] = "Magic Wind";

    // 将 JSON 转换为字符串
    char jsonBuffer[512];
    serializeJson(doc, jsonBuffer);

    // 创建唯一的主题或 object_id
    String uniqueId = "homeassistant/device_automation/" + String("magic_wind/") + action + "/config";
    mqttClient.publish(uniqueId.c_str(), jsonBuffer);  // 使用保留消息
  }
  Serial.println("\nDevice configuration published");
}


void touchCallback() {
  // Serial.println("Touched.");
  // if (isRecording) {
  //   isRecording = false;  // 松开按钮，停止记录
  //   Serial.println("Stopped recording.");
  // } else {
  //   isRecording = true;  // 按下按钮，开始记录
  //   Serial.println("Started recording.");
}

//获取闪存中的WIFI配置
void getWIFIconfig() {
  if (LittleFS.exists(wifi_config_file)) {
    Serial.print(wifi_config_file);
    Serial.println(" FOUND.");
    //建立File对象用于从LittleFS中读取文件
    File dataFile = LittleFS.open(wifi_config_file, "r");
    ssid = (dataFile.readStringUntil('\n'));
    password = (dataFile.readStringUntil('\n'));
    ssid.trim();
    password.trim();
    dataFile.close();
  } else {
    Serial.print(wifi_config_file);
    Serial.println(" NOT FOUND.");
  }
}
// 处理修改WIFI请求
void handleWIFI() {
  //  ssid = server.arg("ssid").c_str();
  //  password = server.arg("password").c_str();

  File configFile = LittleFS.open(wifi_config_file, "w");
  if (configFile) {
    configFile.println(server.arg("ssid"));      // 将你的SSID写入文件
    configFile.println(server.arg("password"));  // 将你的密码写入文件
    configFile.close();
    Serial.println("WIFI config file written successfully");
    server.send(200, "text/html", "Config file written successfully");
  } else {
    Serial.println("Failed to open WIFI config file for writing");
    server.send(400, "text/html", "Failed to open WIFI config file for writing");
  }
}
//获取闪存中的WIFI配置
void getMQTTconfig() {
  if (LittleFS.exists(mqtt_config_file)) {
    Serial.print(mqtt_config_file);
    Serial.println(" FOUND.");
    //建立File对象用于从LittleFS中读取文件
    File dataFile = LittleFS.open(mqtt_config_file, "r");
    mqttServer = (dataFile.readStringUntil('\n'));
    mqttPort = (dataFile.readStringUntil('\n'));
    mqttUser = (dataFile.readStringUntil('\n'));
    mqttPassword = (dataFile.readStringUntil('\n'));
    mqttServer.trim();
    mqttPort.trim();
    mqttUser.trim();
    mqttPassword.trim();
    dataFile.close();
  } else {
    Serial.print(mqtt_config_file);
    Serial.println(" NOT FOUND.");
  }
}
// 处理修改MQTT请求
void handleMQTT() {
  //  mqttServer = server.arg("ssid").c_str();
  //  mqttPort = server.arg("password").c_str();
  //  mqttUser = server.arg("ssid").c_str();
  //  mqttPassword = server.arg("password").c_str();

  File configFile = LittleFS.open(mqtt_config_file, "w");
  if (configFile) {
    configFile.println(server.arg("server"));    // 将你的SSID写入文件
    configFile.println(server.arg("port"));      // 将你的密码写入文件
    configFile.println(server.arg("user"));      // 将你的密码写入文件
    configFile.println(server.arg("password"));  // 将你的密码写入文件
    configFile.close();
    Serial.println("MQTT config file written successfully");
    server.send(200, "text/html", "MQTT config file written successfully");
  } else {
    Serial.println("Failed to open MQTT config file for writing");
    server.send(400, "text/html", "Failed to open MQTT config file for writing");
  }
}
// 处理重启请求
void handleRestart() {
  // 输出调试信息
  Serial.println("Restarting...");
  server.send(200, "text/html", "Restarting...");
  delay(500);
  // 调用 ESP.restart() 重启 ESP32
  ESP.restart();
}
void handleUploadForm() {
  HTTPUpload& upload = server.upload();
  String filename = "/" + upload.filename;  // 文件名
  if (upload.status == UPLOAD_FILE_START) {
    // 如果上传文件开始，打开文件准备写入

    Serial.printf("Uploading File: %s\n", filename.c_str());

    // 如果文件已存在，删除旧文件
    if (LittleFS.exists(filename)) {
      LittleFS.remove(filename);
    }

    // 打开文件进行写入
    File file = LittleFS.open(filename, "w");
    if (!file) {
      Serial.println("Failed to open file for writing");
      return;
    }
    file.close();  // 关闭文件，准备开始写入数据
  }

  if (upload.status == UPLOAD_FILE_WRITE) {
    // 写入上传的数据块
    File file = LittleFS.open(filename, "a");
    if (file) {
      file.write(upload.buf, upload.currentSize);
      file.close();  // 每次写完数据后关闭文件
      Serial.printf("Uploaded %d bytes\n", upload.currentSize);
    }
  }

  if (upload.status == UPLOAD_FILE_END) {
    // 上传结束时
    Serial.printf("File upload complete: %s\n", upload.filename.c_str());
    server.send(200, "text/plain", "Upload complete");
  }
}

void handleFileUpload() {
  // 处理上传完成后的响应，可以返回一个成功消息
  server.send(200, "text/plain", "File uploaded successfully!");
}

void handleUpdateBin() {
    HTTPUpload& upload = server.upload();
    if (upload.status == UPLOAD_FILE_START) {
      // 开始接收上传的文件
      Serial.printf("Update Start: %s\n", upload.filename.c_str());
      if (!Update.begin(UPDATE_SIZE_UNKNOWN)) {
        Update.printError(Serial);
      }
    } else if (upload.status == UPLOAD_FILE_WRITE) {
      // 写入数据
      if (Update.write(upload.buf, upload.currentSize) != upload.currentSize) {
        Update.printError(Serial);
      }
    } else if (upload.status == UPLOAD_FILE_END) {
      // 上传结束，完成固件更新
      if (Update.end(true)) {
        Serial.printf("Update Success: %u bytes\n", upload.totalSize);
        ESP.restart();
      } else {
        Update.printError(Serial);
      }
    }
  }
void handleFirmwareUpload() {
  server.send(200, "text/plain", "Firmware uploaded successfully!");
}
//处理html类请求
void handleUserRequet() {
  // 获取用户请求网址信息
  String webAddress = server.uri();
  // 通过handleFileRead函数处处理用户访问
  bool fileReadOK = handleFileRead(webAddress);
  // 如果在LittleFS无法找到用户访问的资源，则回复404 (Not Found)
  if (!fileReadOK) {
    server.send(404, "text/plain", "404 Not Found");
  }
}
//处理文件请求
bool handleFileRead(String path) {
  // 将访问地址修改为/index.html便于LittleFS访问
  if (path.endsWith("/")) {
    path = "/index.html";
  }
  String contentType = getContentType(path);  // 获取文件类型
  if (LittleFS.exists(path)) {
    //    读取闪存文件并返回
    File file = LittleFS.open(path, "r");
    server.streamFile(file, contentType);
    file.close();
    return true;
  }
  return false;  // 如果文件未找到，则返回false
}
// 获取文件类型
String getContentType(String filename) {
  if (filename.endsWith(".htm")) return "text/html";
  else if (filename.endsWith(".html")) return "text/html";
  else if (filename.endsWith(".css")) return "text/css";
  else if (filename.endsWith(".js")) return "application/javascript";
  else if (filename.endsWith(".png")) return "image/png";
  else if (filename.endsWith(".gif")) return "image/gif";
  else if (filename.endsWith(".jpg")) return "image/jpeg";
  else if (filename.endsWith(".ico")) return "image/x-icon";
  else if (filename.endsWith(".xml")) return "text/xml";
  else if (filename.endsWith(".pdf")) return "application/x-pdf";
  else if (filename.endsWith(".zip")) return "application/x-zip";
  else if (filename.endsWith(".gz")) return "application/x-gzip";
  return "text/plain";
}

// 加载模型权重
void loadModelWeights(const char* filename) {
  File file = LittleFS.open(filename, "r");
  if (!file) {
    Serial.println("Failed to open model-weights file");
    return;
  }

  // 解析 JSON
  StaticJsonDocument<121024> doc;
  DeserializationError error = deserializeJson(doc, file);
  if (error) {
    Serial.println("Failed to read model-weights");
    return;
  }

  // 从 JSON 中加载权重
  for (int i = 0; i < layer2; i++) {
    fc1_bias[i] = doc["fc1.bias"][i];
    for (int j = 0; j < layer1; j++) {
      fc1_weight[i * layer1 + j] = doc["fc1.weight"][i][j];
    }
  }

  for (int i = 0; i < layer3; i++) {
    fc3_bias[i] = doc["fc3.bias"][i];
    for (int j = 0; j < layer2; j++) {
      fc3_weight[i * layer2 + j] = doc["fc3.weight"][i][j];
    }
  }


  file.close();
}

// 加载 OneHotEncoder 类别信息
void loadOneHotEncoder(const char* filename) {
  File file = LittleFS.open(filename, "r");
  if (!file) {
    Serial.println("Failed to open OneHotEncoder file");
    return;
  }

  // 解析 JSON
  StaticJsonDocument<1024> doc;
  DeserializationError error = deserializeJson(doc, file);
  if (error) {
    Serial.println("Failed to read OneHotEncoder");
    return;
  }

  // 清空 actions vector，确保每次加载时都是干净的
  actions.clear();

  // 读取类别信息
  for (JsonVariant category : doc.as<JsonArray>()[0].as<JsonArray>()) {
    actions.push_back(category.as<String>());
  }

  file.close();
}
