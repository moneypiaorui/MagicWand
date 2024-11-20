# Arduino Nano 33 BLE Gesture Classifier

This example provides an updated and working version of the sketch IMU_Classifier presented in the official Arduino tutorial "*Get Started With Machine Learning on Arduino*" that you can [view here](https://docs.arduino.cc/tutorials/nano-33-ble-sense/get-started-with-machine-learning/).

The sketch has been rewritten to work with the updated version of the TensorFlow Lite Micro library through the **ArduTFLite** library APIs. The original tutorial explains the process of collecting training data and how the model was created and trained. The model.h file included in this example was pre-trained by Chirale during the preparation activities for our TinyML workshop on Arduino.

The sketch provided is designed to classify gestures, specifically "punch" and "flex", using the ArduTFLite library and an IMU sensor integrated into the Arduino Nano 33 BLE board. It demonstrates the use of a pre-trained neural network model to recognize gestures based on IMU readings.

### Explanation of the Sketch

The sketch has several key parts, each explained below:

```cpp
/*
  Gesture Classifier
  This sketch demonstrates the use of the ArduTFLite library to recognize the two gestures "punch" and "flex" 
  through readings from the IMU sensor integrated into the Arduino Nano 33 BLE board. 
  This example is an updated variation of the famous tutorial created by Don Coleman, Sandeep Mistry, 
  and Dominic Pajak in 2019. 
  Along with the sketch, the file model.h is included, which contains a pre-trained model.

  The sketch was created and tested on Arduino Nano 33 BLE rev 1 and rev 2 boards, but it can be adapted 
  to run on other boards, compatible with the library, which have an IMU sensor connected with a sampling 
  frequency close to 100 Hz.

  CIRCUIT: Arduino Nano 33 BLE Rev.1 or Rev.2

  USAGE: uncomment IMU sensor library, according to your board revision, compile, load and open serial monitor
         keep the board in your hand and do a gesture!

  
  This example code is in the public domain.
*/
```
### Comments
- **Purpose**: Explains that the sketch classifies "punch" and "flex" gestures using the ArduTFLite library and IMU sensor data.
- **Origin**: Mentions that this is an updated version of a tutorial by Don Coleman, Sandeep Mistry, and Dominic Pajak.
- **Model**: Indicates that the pre-trained model is included in `model.h`.
- **Boards**: Specifies compatibility with Arduino Nano 33 BLE rev 1 and rev 2 and other boards with similar IMU sensors.
- **Usage**: Instructions for selecting the correct IMU library based on board revision and running the sketch.

### IMU Sensor Library

```cpp
// Uncomment library definition, according to your board version
//#include <Arduino_LSM9DS1.h> // IMU Sensor Library for Arduino Nano 33 BLE Rev.1
#include <Arduino_BMI270_BMM150.h> // IMU Sensor Library for Arduino Nano 33 BLE Rev.2
```
- **IMU Libraries**: Depending on the board revision, either `Arduino_LSM9DS1` for rev 1 or `Arduino_BMI270_BMM150` for rev 2 is included.

### Library and Model Includes

```cpp
#include <ArduTFLite.h>
#include "model.h"
```
- **ArduTFLite**: Includes the main ArduTFLite library.
- **Model**: Includes the pre-trained model header file.

### Constants and Variables

```cpp
const float accelerationThreshold = 2.5; // Threshold (in G values) to detect a "gesture" start
const int numSamples = 119; // Number of samples for a single gesture
int samplesRead; // Sample counter 
const int inputLength = 714; // Dimension of input tensor (6 values * 119 samples)
constexpr int tensorArenaSize = 8 * 1024; // Tensor Arena size
alignas(16) byte tensorArena[tensorArenaSize]; // Tensor Arena memory
const char* GESTURES[] = { "punch", "flex" }; // Gesture labels
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0])) // Number of gestures
```
- **accelerationThreshold**: Defines the threshold to detect the start of a gesture.
- **numSamples**: Number of samples taken for a single gesture.
- **samplesRead**: Counter for the number of samples read.
- **inputLength**: Size of the input tensor.
- **tensorArenaSize**: Size of the tensor arena.
- **tensorArena**: Allocates memory for the tensor arena with proper alignment.
- **GESTURES**: Array of gesture labels.
- **NUM_GESTURES**: Calculates the number of gestures.

### Setup Function

```cpp
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // init IMU sensor
  if (!IMU.begin()) {
    Serial.println("IMU sensor init failed!");
    while (true); // stop program here.
  }

  // print IMU sampling frequencies
  Serial.print("Accelerometer sampling frequency = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sampling frequency = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();
  Serial.println("Init model..");
  if (!modelInit(model, tensorArena, tensorArenaSize)){
    Serial.println("Model initialization failed!");
    while(true);
  }
  Serial.println("Model initialization done.");
}
```
- **Serial Communication**: Initializes serial communication and waits for the Serial Monitor to open.
- **IMU Initialization**: Initializes the IMU sensor and prints the sampling frequencies of the accelerometer and gyroscope.
- **Model Initialization**: Initializes the TensorFlow Lite model and prints a success message or stops the program if initialization fails.

### Loop Function

```cpp
void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // wait for a significant movement
  while (true) {
    if (IMU.accelerationAvailable()) {
      // read linear acceleration
      IMU.readAcceleration(aX, aY, aZ);

      // compute absolute value of total acceleration
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // if total absolute acceleration is over the threshold a gesture has started
      if (aSum >= accelerationThreshold) {
        samplesRead = 0; // init samples counter
        break; // exit from waiting cycle
      }
    }
  }

  // reading cycle of all samples for current gesture
  while (samplesRead < numSamples) {
    // check if a sample is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read acceleration and gyroscope values
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // normalize sensor data because model was trained using normalized data
      aX = (aX + 4.0) / 8.0;
      aY = (aY + 4.0) / 8.0;
      aZ = (aZ + 4.0) / 8.0;
      gX = (gX + 2000.0) / 4000.0;
      gY = (gY + 2000.0) / 4000.0;
      gZ = (gZ + 2000.0) / 4000.0;
      
      // put the 6 values of current sample in the proper position
      // in the input tensor of the model
      modelSetInput(aX, samplesRead * 6 + 0);
      modelSetInput(aY, samplesRead * 6 + 1);
      modelSetInput(aZ, samplesRead * 6 + 2); 
      modelSetInput(gX, samplesRead * 6 + 3);
      modelSetInput(gY, samplesRead * 6 + 4);
      modelSetInput(gZ, samplesRead * 6 + 5); 
      
      samplesRead++;
      
      // if all samples are got, run inference
      if (samplesRead == numSamples) {
        if(!modelRunInference()){
          Serial.println("RunInference Failed!");
          return;
        }

        // get output values and print as percentage
        for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.print(modelGetOutput(i)*100, 2);
          Serial.println("%");
        }
        Serial.println();
      }
    }
  }
}
```
- **Waiting for Movement**: Continuously checks for significant movement by reading accelerometer values and computing the absolute value of total acceleration. If the acceleration exceeds the threshold, it starts reading gesture samples.
- **Reading Samples**: Reads accelerometer and gyroscope values, normalizes the data, and stores it in the model's input tensor. Continues reading until the required number of samples is collected.
- **Running Inference**: Once all samples are collected, runs the model inference and prints the output percentages for each gesture.

### Summary

This sketch demonstrates how to use the ArduTFLite library to classify gestures based on IMU sensor data. It initializes the IMU sensor, waits for significant movement, reads a predefined number of samples, normalizes the data, and runs inference using a pre-trained model to classify the gestures "punch" and "flex". The results are printed to the Serial Monitor as percentages for each gesture.
