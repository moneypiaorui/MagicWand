/****************************************************************************************************************************
  defines.h
  For any WiFi shields, such as ESP32, ESP8266, Portenta_H7, WiFiNINA W101, W102, W13x, or custom, such as ESP8266/ESP32-AT, etc
  
  WiFiMulti_Generic is a library to adapt the  ESP32/ESP8266 WiFiMulti feature to other WiFi modules
  
  Based on and modified from WiFiMulti of ESP32 core: https://github.com/espressif/arduino-esp32
  Based on and modified from WiFiMulti of ESP8266 core: https://github.com/esp8266/Arduino
 
  Built by Khoi Hoang https://github.com/khoih-prog/WiFiMulti_Generic
  
  License under GPL-3.0
 ***************************************************************************************************************************************/

#ifndef defines_h
#define defines_h

#if !( defined(ARDUINO_RASPBERRY_PI_PICO_W) )
  #error For RASPBERRY_PI_PICO_W only
#endif

#define DEBUG_WIFI_MULTI_GENERIC_PORT   Serial

// Debug Level from 0 to 4
#define _WIFIMULTI_LOGLEVEL_            3

#define _WIFI_LOGLEVEL_                 1

#define USE_WIFI_NINA         false
#define USE_WIFI101           false
#define USE_WIFI_CUSTOM       false

//#include "WiFi.h" 

#if defined(WIFI_USE_RP2040)
  #undef WIFI_USE_RP2040
#endif

#define WIFI_USE_RP2040      true

#warning RP2040 board selected

#include <WiFiMulti_Generic.h>

char your_ssid[] = "YOUR_SSID";         // your network SSID (name)
char your_pass[] = "12345678";          // your network password
char your_ssid1[] = "YOUR_SSID";        // your network SSID (name)
char your_pass1[] = "12345678";         // your network password

#endif    //defines_h
