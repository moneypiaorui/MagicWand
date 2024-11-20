/**************************************************************************************************************************************
  RP2040W_WiFiMulti_Generic.hpp
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

#ifndef _RP2040W_WIFIMULTI_GENERIC_HPP_
#define _RP2040W_WIFIMULTI_GENERIC_HPP_

#include <vector>

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
#endif /* _RP2040W_WIFIMULTI_GENERIC_HPP_ */
