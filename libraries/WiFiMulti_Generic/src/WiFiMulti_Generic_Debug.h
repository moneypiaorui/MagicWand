/**************************************************************************************************************************************
  WiFiMulti_Generic_Debug.h
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

#ifndef _WIFIMULTI_GENERIC_DEBUG_H_
#define _WIFIMULTI_GENERIC_DEBUG_H_

#if defined(ARDUINO)
  #if ARDUINO >= 100
    #include <Arduino.h>
  #else
    #include <WProgram.h>
  #endif
#endif

#include <stdio.h>

#ifdef DEBUG_WIFI_WEBSERVER_PORT
  #define WFM_DEBUG_OUTPUT DEBUG_WIFI_WEBSERVER_PORT
#else
  #define WFM_DEBUG_OUTPUT Serial
#endif

// Change _WIFIMULTI_LOGLEVEL_ to set tracing and logging verbosity
// 0: DISABLED: no logging
// 1: ERROR: errors
// 2: WARN: errors and warnings
// 3: INFO: errors, warnings and informational (default)
// 4: DEBUG: errors, warnings, informational and debug

#ifndef _WIFIMULTI_LOGLEVEL_
  #define _WIFIMULTI_LOGLEVEL_       0
#endif

const char WFM_MARK[]  = "[WFM] ";
const char WFM_SPACE[] = " ";
const char WFM_LINE[]  = "==============================================================\n";

#define WFM_PRINT_MARK      WFM_PRINT(WFM_MARK)
#define WFM_PRINT_SP        WFM_PRINT(WFM_SPACE)

#if(_WIFIMULTI_LOGLEVEL_> 3)
  #define WFM_PRINT_LINE      WFM_PRINT(WFM_LINE)
#else
  #define WFM_PRINT_LINE
#endif  

#define WFM_PRINT           WFM_DEBUG_OUTPUT.print
#define WFM_PRINTLN         WFM_DEBUG_OUTPUT.println

#define WFM_PRINT_HEX(x)    WFM_PRINT(x, HEX)
#define WFM_PRINTLN_HEX(x)  WFM_PRINTLN(x, HEX)

///////////////////////////////////////

#define WFM_LOGERROR(x)         if(_WIFIMULTI_LOGLEVEL_>0) { WFM_PRINT_MARK; WFM_PRINTLN(x); }
#define WFM_LOGERROR0(x)        if(_WIFIMULTI_LOGLEVEL_>0) { WFM_PRINT(x); }
#define WFM_LOGERROR1(x,y)      if(_WIFIMULTI_LOGLEVEL_>0) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINTLN(y); }
#define WFM_LOGERROR2(x,y,z)    if(_WIFIMULTI_LOGLEVEL_>0) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINTLN(z); }
#define WFM_LOGERROR3(x,y,z,w)  if(_WIFIMULTI_LOGLEVEL_>0) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINT(z); WFM_PRINT_SP; WFM_PRINTLN(w); }
#define WFM_LOGERROR5(x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>0) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINT(z); WFM_PRINT_SP; WFM_PRINT(w); WFM_PRINT_SP; WFM_PRINT(xx); WFM_PRINT_SP; WFM_PRINTLN(yy); }

#define WFM_LOGERROR5_SEPARATOR(s,x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>0) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT(s); WFM_PRINT(y); WFM_PRINT(s); WFM_PRINT(z); WFM_PRINT(s); WFM_PRINT(w); WFM_PRINT(s); WFM_PRINT(xx); WFM_PRINT(s); WFM_PRINTLN(yy); }

#define WFM_HEXLOGERROR5_SEPARATOR(s,x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>0) { WFM_PRINT_MARK; WFM_PRINT_HEX(x); WFM_PRINT(s); WFM_PRINT_HEX(y); WFM_PRINT(s); WFM_PRINT_HEX(z); WFM_PRINT(s); WFM_PRINT_HEX(w); WFM_PRINT(s); WFM_PRINT_HEX(xx); WFM_PRINT(s); WFM_PRINTLN_HEX(yy); }

///////////////////////////////////////

#define WFM_LOGWARN(x)          if(_WIFIMULTI_LOGLEVEL_>1) { WFM_PRINT_MARK; WFM_PRINTLN(x); }
#define WFM_LOGWARN0(x)         if(_WIFIMULTI_LOGLEVEL_>1) { WFM_PRINT(x); }
#define WFM_LOGWARN1(x,y)       if(_WIFIMULTI_LOGLEVEL_>1) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINTLN(y); }
#define WFM_LOGWARN2(x,y,z)     if(_WIFIMULTI_LOGLEVEL_>1) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINTLN(z); }
#define WFM_LOGWARN3(x,y,z,w)   if(_WIFIMULTI_LOGLEVEL_>1) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINT(z); WFM_PRINT_SP; WFM_PRINTLN(w); }
#define WFM_LOGWARN5(x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>1) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINT(z); WFM_PRINT_SP; WFM_PRINT(w); WFM_PRINT_SP; WFM_PRINT(xx); WFM_PRINT_SP; WFM_PRINTLN(yy); }

#define WFM_LOGWARN5_SEPARATOR(s,x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>1) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT(s); WFM_PRINT(y); WFM_PRINT(s); WFM_PRINT(z); WFM_PRINT(s); WFM_PRINT(w); WFM_PRINT(s); WFM_PRINT(xx); WFM_PRINT(s); WFM_PRINTLN(yy); }

#define WFM_HEXLOGWARN5_SEPARATOR(s,x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>1) { WFM_PRINT_MARK; WFM_PRINT_HEX(x); WFM_PRINT(s); WFM_PRINT_HEX(y); WFM_PRINT(s); WFM_PRINT_HEX(z); WFM_PRINT(s); WFM_PRINT_HEX(w); WFM_PRINT(s); WFM_PRINT_HEX(xx); WFM_PRINT(s); WFM_PRINTLN_HEX(yy); }

///////////////////////////////////////

#define WFM_LOGINFO(x)          if(_WIFIMULTI_LOGLEVEL_>2) { WFM_PRINT_MARK; WFM_PRINTLN(x); }
#define WFM_LOGINFO0(x)         if(_WIFIMULTI_LOGLEVEL_>2) { WFM_PRINT(x); }
#define WFM_LOGINFO1(x,y)       if(_WIFIMULTI_LOGLEVEL_>2) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINTLN(y); }
#define WFM_LOGINFO2(x,y,z)     if(_WIFIMULTI_LOGLEVEL_>2) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINTLN(z); }
#define WFM_LOGINFO3(x,y,z,w)   if(_WIFIMULTI_LOGLEVEL_>2) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINT(z); WFM_PRINT_SP; WFM_PRINTLN(w); }

#define WFM_LOGINFO5(x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>2) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINT(z); WFM_PRINT_SP; WFM_PRINT(w); WFM_PRINT_SP; WFM_PRINT(xx); WFM_PRINT_SP; WFM_PRINTLN(yy); }

#define WFM_LOGINFO5_SEPARATOR(s,x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>2) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT(s); WFM_PRINT(y); WFM_PRINT(s); WFM_PRINT(z); WFM_PRINT(s); WFM_PRINT(w); WFM_PRINT(s); WFM_PRINT(xx); WFM_PRINT(s); WFM_PRINTLN(yy); }

#define WFM_HEXLOGINFO5_SEPARATOR(s,x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>2) { WFM_PRINT_MARK; WFM_PRINT_HEX(x); WFM_PRINT(s); WFM_PRINT_HEX(y); WFM_PRINT(s); WFM_PRINT_HEX(z); WFM_PRINT(s); WFM_PRINT_HEX(w); WFM_PRINT(s); WFM_PRINT_HEX(xx); WFM_PRINT(s); WFM_PRINTLN_HEX(yy); }

///////////////////////////////////////

#define WFM_LOGDEBUG(x)         if(_WIFIMULTI_LOGLEVEL_>3) { WFM_PRINT_MARK; WFM_PRINTLN(x); }
#define WFM_LOGDEBUG0(x)        if(_WIFIMULTI_LOGLEVEL_>3) { WFM_PRINT(x); }
#define WFM_LOGDEBUG1(x,y)      if(_WIFIMULTI_LOGLEVEL_>3) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINTLN(y); }
#define WFM_LOGDEBUG2(x,y,z)    if(_WIFIMULTI_LOGLEVEL_>3) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINTLN(z); }
#define WFM_LOGDEBUG3(x,y,z,w)  if(_WIFIMULTI_LOGLEVEL_>3) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINT(z); WFM_PRINT_SP; WFM_PRINTLN(w); }
#define WFM_LOGDEBUG5(x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>3) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT_SP; WFM_PRINT(y); WFM_PRINT_SP; WFM_PRINT(z); WFM_PRINT_SP; WFM_PRINT(w); WFM_PRINT_SP; WFM_PRINT(xx); WFM_PRINT_SP; WFM_PRINTLN(yy); }

#define WFM_LOGDEBUG5_SEPARATOR(s,x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>3) { WFM_PRINT_MARK; WFM_PRINT(x); WFM_PRINT(s); WFM_PRINT(y); WFM_PRINT(s); WFM_PRINT(z); WFM_PRINT(s); WFM_PRINT(w); WFM_PRINT(s); WFM_PRINT(xx); WFM_PRINT(s); WFM_PRINTLN(yy); }

#define WFM_HEXLOGDEBUG5_SEPARATOR(s,x,y,z,w,xx,yy) if(_WIFIMULTI_LOGLEVEL_>3) { WFM_PRINT_MARK; WFM_PRINT_HEX(x); WFM_PRINT(s); WFM_PRINT_HEX(y); WFM_PRINT(s); WFM_PRINT_HEX(z); WFM_PRINT(s); WFM_PRINT_HEX(w); WFM_PRINT(s); WFM_PRINT_HEX(xx); WFM_PRINT(s); WFM_PRINTLN_HEX(yy); }

///////////////////////////////////////

#endif    // _WIFIMULTI_GENERIC_DEBUG_H_
