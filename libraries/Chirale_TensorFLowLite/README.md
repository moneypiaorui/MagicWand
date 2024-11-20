
# TensorFlow Lite Micro Library for Arduino

## Overview

This is a port of the TensorFlow Lite Micro Library to the Arduino platform, aimed at enabling Tiny Machine Learning (TinyML) experiments on all Arduino boards with mbed or ESP32 architecture.

## Compatibility

- **Tested Boards**: Arduino Nano 33 BLE, Arduino Nano ESP32, Arduino Nicla, Arduino Portenta, Arduino Giga R1 WiFi
- **Other Boards**: Compatibility with other Arduino boards is not guaranteed and must be tested.

## Resources

The Arduino boards with mbed or ESP32 architecture have sufficient resources to host TensorFlow Lite (TFLite) models.

## Experimental Nature

Please note that this library is experimental and has some differences from the typical Arduino library style.
It is similar to older and outdated Arduino_TensorFlowLite Micro example library, published in 2019 and referred by Arduino tutorials, but It has been regenerated from main TFLite Micro Library and is updated.

## Are You Looking for a more "Arduino Style" Library?

**Chirale** has developed a companion library, called ***ArduTFLite***, that has simpler APIs in Arduino Style. For learning purposes and for a quick introduction to Tiny Machine Learning we recommend to install and use *ArduTFLite*.

## Features

- **TensorFlow Lite Interpreter**: Allows instantiation of a TensorFlow Lite interpreter.
- **Memory Allocation**: Allocates space for the Tensor Arena and for input and output tensors.
- **Inference Operations**: Enables the execution of inference operations.

## Getting Started

For more details on how to use this library, refer to the `hello_world` example included in the repository.


