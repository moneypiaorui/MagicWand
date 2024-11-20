/**************************************************************************************************************************************
  WiFiMulti_Generic.h
  For any WiFi shields, such as ESP32, ESP8266, Portenta_H7, WiFiNINA W101, W102, W13x, or custom, such as ESP8266/ESP32-AT, etc
  
  WiFiMulti_Generic is a library to adapt the  ESP32/ESP8266 WiFiMulti feature to other WiFi modules
  
  Based on and modified from WiFiMulti of ESP32 core: https://github.com/espressif/arduino-esp32
  Based on and modified from WiFiMulti of ESP8266 core: https://github.com/esp8266/Arduino
 
  Built by Khoi Hoang https://github.com/khoih-prog/WiFiMulti_Generic
  
  License under GPL-3.0
  
  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
  as published bythe Free Software Foundation, either version 3 of the License, or (at your option) any later version.
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
  You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>
  
  Version: 1.2.2
  
  Version Modified By   Date      Comments
  ------- -----------  ---------- -----------
  1.0.0   K Hoang      15/02/2020 Initial coding for ESP32, ESP8266, WiFiNINA and ESP_AT modules
  1.1.0   K Hoang      16/02/2020 Add support to WiFi101 and many more boards (PortentaH7, megaAVR, Sparkfun SAMD, etc.)
  1.1.1   K Hoang      26/04/2020 Fix bug
  1.2.0   K Hoang      12/08/2022 Add support to RASPBERRY_PI_PICO_W using CYW4343 WiFi
  1.2.1   K Hoang      16/08/2022 Workaround for RP2040W WiFi.status() bug
  1.2.2   K Hoang      16/08/2022 Better workaround for RP2040W WiFi.status() bug using ping() to local gateway
 ***************************************************************************************************************************************/

#pragma once

#ifndef _WIFIMULTI_GENERIC_H_
#define _WIFIMULTI_GENERIC_H_

#ifndef WIFIMULTI_GENERIC_VERSION
  #define WIFIMULTI_GENERIC_VERSION          "WiFiMulti_Generic v1.2.2"

  #define WIFIMULTI_GENERIC_VERSION_MAJOR    1
  #define WIFIMULTI_GENERIC_VERSION_MINOR    2
  #define WIFIMULTI_GENERIC_VERSION_PATCH    2

  #define WIFIMULTI_GENERIC_VERSION_INT      1002002
#endif

#if defined(ARDUINO)
  #if ARDUINO >= 100
    #include <Arduino.h>
  #else
    #include <WProgram.h>
  #endif
#endif

#undef max
#undef min

#if !defined(BOARD_NAME)
  #if defined(ARDUINO_BOARD)
    // For ESP32/ESP8266 or Arduino boards
    #define BOARD_NAME      ARDUINO_BOARD
  #endif
#endif


#if ( defined(ARDUINO_PORTENTA_H7_M7) || defined(ARDUINO_PORTENTA_H7_M4) )
  #if defined(WIFI_USE_PORTENTA_H7)
    #undef WIFI_USE_PORTENTA_H7
  #endif
  #define WIFI_USE_PORTENTA_H7        true
  
  #if defined(USE_NEW_WEBSERVER_VERSION)
    #undef USE_NEW_WEBSERVER_VERSION
  #endif
  
  // KH, comment out from v1.2.0
  //#define USE_NEW_WEBSERVER_VERSION   false
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use mbed-portenta architecture for PORTENTA_H7 from WiFiMulti_Generic
  #endif
#endif

#if    ( defined(ARDUINO_SAMD_ZERO) || defined(ARDUINO_SAMD_MKR1000) || defined(ARDUINO_SAMD_MKRWIFI1010) \
      || defined(ARDUINO_SAMD_NANO_33_IOT) || defined(ARDUINO_SAMD_MKRFox1200) || defined(ARDUINO_SAMD_MKRWAN1300) || defined(ARDUINO_SAMD_MKRWAN1310) \
      || defined(ARDUINO_SAMD_MKRGSM1400) || defined(ARDUINO_SAMD_MKRNB1500) || defined(ARDUINO_SAMD_MKRVIDOR4000) || defined(__SAMD21G18A__) \
      || defined(ARDUINO_SAMD_CIRCUITPLAYGROUND_EXPRESS) || defined(__SAMD21E18A__) || defined(__SAMD51__) || defined(__SAMD51J20A__) || defined(__SAMD51J19A__) \
      || defined(__SAMD51G19A__) || defined(__SAMD51P19A__) || defined(__SAMD21G18A__) )
  #if defined(WIFI_USE_SAMD)
    #undef WIFI_USE_SAMD
  #endif
  #define WIFI_USE_SAMD      true
  
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use SAMD architecture from WiFiMulti_Generic
  #endif
#endif

#if (defined(NRF52840_FEATHER) || defined(NRF52832_FEATHER) || defined(NRF52_SERIES) || defined(ARDUINO_NRF52_ADAFRUIT) || \
     defined(NRF52840_FEATHER_SENSE) || defined(NRF52840_ITSYBITSY) || defined(NRF52840_CIRCUITPLAY) || \
     defined(NRF52840_CLUE) || defined(NRF52840_METRO) || defined(NRF52840_PCA10056) || defined(PARTICLE_XENON) || \
     defined(MDBT50Q_RX) || defined(NINA_B302_ublox) || defined(NINA_B112_ublox) )
  #if defined(WIFI_USE_NRF528XX)
    #undef WIFI_USE_NRF528XX
  #endif
  #define WIFI_USE_NRF528XX      true
  
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use nFR52 architecture from WiFiMulti_Generic
  #endif
  
  #include <Adafruit_TinyUSB.h>

#endif

#if ( defined(ARDUINO_SAM_DUE) || defined(__SAM3X8E__) )
  #if defined(WIFI_USE_SAM_DUE)
    #undef WIFI_USE_SAM_DUE
  #endif
  #define WIFI_USE_SAM_DUE      true
  
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use SAM_DUE architecture from WiFiMulti_Generic
  #endif
#endif

#if ( defined(STM32F0) || defined(STM32F1) || defined(STM32F2) || defined(STM32F3)  ||defined(STM32F4) || defined(STM32F7) || \
       defined(STM32L0) || defined(STM32L1) || defined(STM32L4) || defined(STM32H7)  ||defined(STM32G0) || defined(STM32G4) || \
       defined(STM32WB) || defined(STM32MP1) ) && !( defined(ARDUINO_PORTENTA_H7_M7) || defined(ARDUINO_PORTENTA_H7_M4) )

  #if defined(WIFI_USE_STM32)
    #undef WIFI_USE_STM32
  #endif
  #define WIFI_USE_STM32      true
  
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use STM32 architecture from WiFiMulti_Generic
  #endif
#endif

#if ( defined(ARDUINO_NANO_RP2040_CONNECT) || defined(ARDUINO_ARCH_RP2040) || defined(ARDUINO_RASPBERRY_PI_PICO) || \
      defined(ARDUINO_ADAFRUIT_FEATHER_RP2040) || defined(ARDUINO_GENERIC_RP2040) || defined(ARDUINO_RASPBERRY_PI_PICO_W) )
  
  #if (_WIFIMULTI_LOGLEVEL_ > 3)
    #if defined(ARDUINO_RASPBERRY_PI_PICO_W)
      #warning RASPBERRY_PI_PICO_W board using CYW43439 WiFi selected
    #else
      #warning RP2040-based board selected
    #endif
  #endif
  
  #if defined(ARDUINO_RASPBERRY_PI_PICO_W)
    #define WIFI_USE_RP2040W      true
  #else 
    #if defined(WIFI_USE_RP2040)
      #undef WIFI_USE_RP2040
    #endif
    #define WIFI_USE_RP2040      true 
  #endif
  
  #if (_WIFIMULTI_LOGLEVEL_ > 3)
    #warning Use RP2040 architecture from WiFiMulti_Generic
  #endif

#endif

/////////////////////////////////////////////////////////////////

// To suppress warning from some cores, such as AmebaD
#if !defined(USE_WIFI_ESP_AT)
  #define USE_WIFI_ESP_AT       false
#endif

#if !defined(WIFI_USING_ESP_AT)
  #define WIFI_USING_ESP_AT       false
#endif

#if !defined(USE_WIFI_PORTENTA_H7)
  #define USE_WIFI_PORTENTA_H7       false
#endif

/////////////////////////////////////////////////////////////////

#if USE_WIFI_NINA
  #include <WiFiNINA_Generic.h>
  //#include <WiFiNINA.h>
  
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use WiFiNINA from WiFiMulti_Generic
  #endif
#elif USE_WIFI101
  #include <WiFi101.h>
  
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use WiFi101 from WiFiMulti_Generic
  #endif
#elif (USE_WIFI_ESP_AT || WIFI_USING_ESP_AT)
  #if WIFI_USING_ESP8266_AT_WEBSERVER
    #include <ESP8266_AT_WebServer.h>
    
    #if(_WIFIMULTI_LOGLEVEL_> 3)
      #warning Use ESP8266_AT_WebServer Library for WiFiMulti_Generic
    #endif
  #else
    #include <WiFiEspAT.h>
    
    #if(_WIFIMULTI_LOGLEVEL_> 3)
      #warning Use WiFiEspAT Library for WiFiMulti_Generic
    #endif
  #endif
#elif defined(USE_WIFI_PORTENTA_H7) && USE_WIFI_PORTENTA_H7
  #include <WiFi.h>
  
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use PORTENTA_H7 WiFi from WiFiMulti_Generic
  #endif
#elif defined(WIFI_USE_RP2040W) && WIFI_USE_RP2040W
  #include <WiFi.h>
  
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use RP2040W WiFi from WiFiMulti_Generic  
  #endif
#elif defined(CONFIG_PLATFORM_8721D)
  #include <WiFi.h>
  
  #if(_WIFIMULTI_LOGLEVEL_> 3)
    #warning Use RTL8720 WiFi from WiFiMulti_Generic
  #endif
#else
  #if (ESP32)
    #include "WiFi.h"
    
    #if(_WIFIMULTI_LOGLEVEL_> 3)
      #warning Use ESP32 WiFi for WiFiMulti_Generic
    #endif
  #elif (ESP8266)
    #include "ESP8266WiFi.h"
    
    #if(_WIFIMULTI_LOGLEVEL_> 3)
      #warning Use ESP8266 WiFi for WiFiMulti_Generic
    #endif
  #elif (USE_WIFI_CUSTOM && WIFI_USING_ESP_AT)
    #if WIFI_USING_ESP8266_AT_WEBSERVER
      #include <ESP8266_AT_WebServer.h>
      
      #if(_WIFIMULTI_LOGLEVEL_> 3)
        #warning Use ESP8266_AT_WebServer Library for WiFiMulti_Generic
      #endif
    #else
      #include <WiFiEspAT.h>
      
      #if(_WIFIMULTI_LOGLEVEL_> 3)
        #warning Use WiFiEspAT Library for WiFiMulti_Generic
      #endif
    #endif
  #else
    #include <WiFi.h>
    
    #if(_WIFIMULTI_LOGLEVEL_> 3)
      #warning Use WiFi.h from WiFiMulti_Generic
    #endif
  #endif
#endif  

#if defined(ESP32)
  #include "ESP32/WiFiMulti_Generic.hpp"
  #include "ESP32/WiFiMulti_Generic_Impl.h"
#elif defined(ESP8266)
  #include "ESP8266WiFi.h"
  #include "ESP8266/ESP8266WiFiMulti.hpp"
  #include "ESP8266/ESP8266WiFiMulti_Impl.h"
#elif (USE_WIFI_NINA)
  #include "WiFiNINA/NINA_WiFiMulti_Generic.hpp"
  #include "WiFiNINA/NINA_WiFiMulti_Generic_Impl.h"
#elif (USE_WIFI101)
  #include "WiFi101/WiFi101_WiFiMulti_Generic.hpp"
  #include "WiFi101/WiFi101_WiFiMulti_Generic_Impl.h" 
#elif (USE_WIFI_ESP_AT || WIFI_USING_ESP_AT)
  #include "ESP_AT/ESP_AT_WiFiMulti_Generic.hpp"
  #include "ESP_AT/ESP_AT_WiFiMulti_Generic_Impl.h" 
#elif defined(USE_WIFI_PORTENTA_H7) && USE_WIFI_PORTENTA_H7
  #include "PortentaH7/PortentaH7_WiFiMulti_Generic.hpp"
  #include "PortentaH7/PortentaH7_WiFiMulti_Generic_Impl.h" 
#elif defined(WIFI_USE_RP2040W) && WIFI_USE_RP2040W
  #include "RP2040W/RP2040W_WiFiMulti_Generic.hpp"
  #include "RP2040W/RP2040W_WiFiMulti_Generic_Impl.h"   
#elif defined(CONFIG_PLATFORM_8721D)
  #include "RTL8720/RTL8720_WiFiMulti_Generic.hpp"
  #include "RTL8720/RTL8720_WiFiMulti_Generic_Impl.h"
#else
  #error Must specify WiFi type  
#endif

#endif /* _WIFIMULTI_GENERIC_H_ */
