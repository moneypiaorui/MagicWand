/**************************************************************************************************************************************
  ESP_AT_WiFiMulti_Generic.hpp
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
  
  Original author:
  @file ESP8266WiFiMulti.h
  @date 16.05.2015
  @author Markus Sattler
  Copyright (c) 2015 Markus Sattler. All rights reserved.
  This file is part of the esp8266 core for Arduino environment.
  
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

#ifndef _ESP_AT_WIFIMULTI_GENERIC_HPP_
#define _ESP_AT_WIFIMULTI_GENERIC_HPP_

#if ( defined(__AVR_ATmega2560__) || defined(__AVR_ATmega2561__) || defined(__AVR_ATmega1280__) || defined(__AVR_ATmega1281__) || \
      defined(__AVR_ATmega640__) || defined(__AVR_ATmega641__))
  #define USE_ARDUINO_STL           true   
  #define BOARD_TYPE    "Arduino AVR Mega2560/ADK"
  #warning Using Arduino AVR Mega, Mega640(P), Mega2560/ADK.
  
#elif ( defined(__AVR_ATmega644__) || defined(__AVR_ATmega644A__) || defined(__AVR_ATmega644P__) || defined(__AVR_ATmega644PA__)  || \
        defined(ARDUINO_AVR_UNO) || defined(ARDUINO_AVR_NANO) || defined(ARDUINO_AVR_MINI) ||    defined(ARDUINO_AVR_ETHERNET) || \
        defined(ARDUINO_AVR_FIO) || defined(ARDUINO_AVR_BT)   || defined(ARDUINO_AVR_LILYPAD) || defined(ARDUINO_AVR_PRO)      || \
        defined(ARDUINO_AVR_NG) || defined(ARDUINO_AVR_UNO_WIFI_DEV_ED) || defined(ARDUINO_AVR_DUEMILANOVE) )
  #define USE_ARDUINO_STL           true     
  #define BOARD_TYPE    "Arduino AVR UNO, Nano, etc." 
  #warning Using Aduino AVR ATMega644(P), ATMega328(P) such as UNO, Nano.

#elif ( defined(ARDUINO_AVR_FEATHER328P) || defined(ARDUINO_AVR_METRO) || defined(ARDUINO_AVR_PROTRINKET5) || defined(ARDUINO_AVR_PROTRINKET3) || \
      defined(ARDUINO_AVR_PROTRINKET5FTDI) || defined(ARDUINO_AVR_PROTRINKET3FTDI) )
  #define USE_ARDUINO_STL           true     
  #define BOARD_TYPE    "Adafruit AVR ATMega328(P)"
  #warning Using Adafruit ATMega328(P), such as AVR_FEATHER328P or AVR_METRO.
        
#elif ( defined(ARDUINO_AVR_LEONARDO) || defined(ARDUINO_AVR_LEONARDO_ETH) || defined(ARDUINO_AVR_YUN) || defined(ARDUINO_AVR_MICRO) || \
        defined(ARDUINO_AVR_ESPLORA)  || defined(ARDUINO_AVR_LILYPAD_USB)  || defined(ARDUINO_AVR_ROBOT_CONTROL) || defined(ARDUINO_AVR_ROBOT_MOTOR) || \
        defined(ARDUINO_AVR_CIRCUITPLAY)  || defined(ARDUINO_AVR_YUNMINI) || defined(ARDUINO_AVR_INDUSTRIAL101) || defined(ARDUINO_AVR_LININO_ONE) )
  #define USE_ARDUINO_STL           true
  #define BOARD_TYPE    "Arduino AVR ATMega32U4"
  #warning Using Arduino ATMega32U4, such as Leonardo or Leonardo ETH.
  
#elif ( defined(ARDUINO_AVR_FLORA8 ) || defined(ARDUINO_AVR_FEATHER32U4) || defined(ARDUINO_AVR_CIRCUITPLAY) || defined(ARDUINO_AVR_ITSYBITSY32U4_5V) || \
        defined(ARDUINO_AVR_ITSYBITSY32U4_3V)  || defined(ARDUINO_AVR_BLUEFRUITMICRO) || defined(ARDUINO_AVR_ADAFRUIT32U4) )
  #define USE_ARDUINO_STL           true
  #define BOARD_TYPE    "Adafruit AVR ATMega32U4"
  #warning Using Adafruit ATMega32U4, such as Feather_32u4, AVR_CIRCUITPLAY, etc.

#elif ( defined(__AVR_ATmega32U4__) || defined(ARDUINO_AVR_MAKEYMAKEY ) || defined(ARDUINO_AVR_PROMICRO) || defined(ARDUINO_AVR_FIOV3) || \
        defined(ARDUINO_AVR_QDUINOMINI) || defined(ARDUINO_AVR_LILYPAD_ARDUINO_USB_PLUS_BOARD ) )
  #define USE_ARDUINO_STL           true
  #define BOARD_TYPE    "Generic or Sparkfun AVR ATMega32U4"
  #warning Using Generic ATMega32U4, such as Sparkfun AVR_MAKEYMAKEY, AVR_PROMICRO, etc.

#elif ( defined(__AVR_ATmega328P__) || defined(ARDUINO_AVR_DIGITAL_SANDBOX ) || defined(ARDUINO_REDBOT) || defined(ARDUINO_AVR_SERIAL_7_SEGMENT) )
  #define USE_ARDUINO_STL           true
  #define BOARD_TYPE    "Generic or Sparkfun AVR ATMega328P"
  #warning Using Generic ATMega328P, such as Sparkfun AVR_DIGITAL_SANDBOX, REDBOT, etc.

#elif ( defined(__AVR_ATmega128RFA1__) || defined(ARDUINO_ATMEGA128RFA1_DEV_BOARD) )
  #define USE_ARDUINO_STL           true
  #define BOARD_TYPE    "Generic or Sparkfun AVR ATMega128RFA1"
  #warning Using Generic ATMega128RFA1, such as Sparkfun ATMEGA128RFA1_DEV_BOARD, etc.
  
#elif ( defined(ARDUINO_AVR_GEMMA) || defined(ARDUINO_AVR_TRINKET3) || defined(ARDUINO_AVR_TRINKET5) )
  #define USE_ARDUINO_STL           true
  #error These AVR boards are not supported! Please check your Tools->Board setting.
     
#endif 

#if ( USE_ARDUINO_STL || defined(ARDUINO_AVR_UNO_WIFI_REV2) || defined(ARDUINO_AVR_NANO_EVERY ) )
  #warning Requiring ArduinoSTL library with ArduinoSTL Library_Patches. Not using <vector>
  #include "ArduinoSTL.h"       // https://github.com/mike-matera/ArduinoSTL
// For megaAVR board using MegaCoreX
#elif ( defined(__AVR_ATmega4809__) || \
      defined(ARDUINO_AVR_ATmega4809) || defined(ARDUINO_AVR_ATmega4808) || defined(ARDUINO_AVR_ATmega3209) || \
      defined(ARDUINO_AVR_ATmega3208) || defined(ARDUINO_AVR_ATmega1609) || defined(ARDUINO_AVR_ATmega1608) || \
      defined(ARDUINO_AVR_ATmega809) || defined(ARDUINO_AVR_ATmega808) ) 
  #error MegaCoreX megaAVR board not supported
  #define USE_ARDUINO_STL           true  
#else
  #warning Not requiring ArduinoSTL library. using <vector>
  #include <vector>
#endif  

#include "WiFiMulti_Generic_Debug.h"

typedef struct
{
  char * ssid;
  char * passphrase;
} WifiAPlist_t;

class WiFiMulti_Generic
{
  public:
    WiFiMulti_Generic();
    ~WiFiMulti_Generic();

    bool addAP(const char* ssid, const char *passphrase = NULL);

    uint8_t run(const uint32_t& connectTimeout = 5000);

  private:
    std::vector<WifiAPlist_t> APlist;
};

#endif /* _ESP_AT_WIFIMULTI_GENERIC_HPP_ */
