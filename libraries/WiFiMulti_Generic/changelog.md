## WiFiMulti_Generic

[![arduino-library-badge](https://www.ardu-badge.com/badge/WiFiMulti_Generic.svg?)](https://www.ardu-badge.com/WiFiMulti_Generic)
[![GitHub release](https://img.shields.io/github/release/khoih-prog/WiFiMulti_Generic.svg)](https://github.com/khoih-prog/WiFiMulti_Generic/releases)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)
[![GitHub issues](https://img.shields.io/github/issues/khoih-prog/WiFiMulti_Generic.svg)](http://github.com/khoih-prog/WiFiMulti_Generic/issues)

---
---

## Table of Contents


* [Changelog](#changelog)
  * [Releases v1.2.2](#releases-v122)
  * [Releases v1.2.1](#releases-v121)
  * [Releases v1.2.0](#releases-v120)
  * [Releases v1.1.1](#releases-v111)
  * [Releases v1.1.0](#releases-v110)
  * [Releases v1.0.0](#releases-v100)

---
---

## Changelog

#### Releases v1.2.2

1.  Better workaround for RP2040W WiFi.status() bug using ping() to local gateway. Check [WiFi.status() wrongly reports WL_CONNECTED even when WiFi is lost and RSSI is always 0 dBm #762](https://github.com/earlephilhower/arduino-pico/issues/762)

#### Releases v1.2.1

1. Workaround for RP2040W WiFi.status() bug. Check [WiFi.status() wrongly reports WL_CONNECTED even when WiFi is lost and RSSI is always 0 dBm #762](https://github.com/earlephilhower/arduino-pico/issues/762)
2. Add example [WiFiMulti_RP2040W](examples/WiFiMulti_RP2040W)

#### Releases v1.2.0

1. Add support to RASPBERRY_PI_PICO_W using CYW43439 WiFi
2. Update `Packages_Patches`

#### Releases v1.1.1

1. Fix bug
2. Update `Packages_Patches`

#### Releases v1.1.0

1. Add support to 

  - **Portenta_H7 using Murata WiFi**
  - **RTL8720DN using built-in 2.4GHz or 5GHz WiFi** with Realtek `AmebaD` core
  - Arduino SAMD using `WiFi101` (MKR1000, MKR1010, etc.)
  - Seeed XIAO RP2020, SAMD using `WiFi101`, `WiFiNINA_Generic` or `WiFiEspAT`
  - Arduino megaAVR boards with Arduino core (UNO WiFi Rev2 using `WiFi101`, Nano Every using `WiFi101`, `WiFiNINA_Generic` or `WiFiEspAT`)
  - CO2 Ampel SAMD board using `WiFi101`, `WiFiNINA_Generic` or `WiFiEspAT`
  - STM32 boards with `STM32duino Maple` core using `WiFi101`, `WiFiNINA_Generic` or `WiFiEspAT`
  - Sparkfun SAMD board using `WiFi101`, `WiFiNINA_Generic` or `WiFiEspAT`
  - Industruino SAMD using `WiFi101`, `WiFiNINA_Generic` or `WiFiEspAT`
  - Tlera Corp STM32WB boards (Firefly-WB55RG, Nucleo-WB55RG, etc.) using `WiFi101`, `WiFiNINA_Generic` or `WiFiEspAT`
  - Maixduino boards using `WiFi101`, `WiFiNINA_Generic` or `WiFiEspAT`
  - Arduino, Sparkfun, Adafruit, etc. AVR boards (Mega, 32U4, etc.). To use `LibraryPatches` for `ArduinoSTL` library

2. Update example [WiFiMulti](examples/WiFiMulti) to handle more complex cases.


#### Releases v1.0.0

1. Initial coding to support **ESP32, ESP8266, WiFiNINA and ESP8266_AT and ESP32_AT WiFi** for many boards (nRF52, SAMD, Teensy, RP2040, SAM-DUE, ESP32, ESP8266, etc.)



