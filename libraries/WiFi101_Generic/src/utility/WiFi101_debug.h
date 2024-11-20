/**********************************************************************************************************************************
  WiFi101_debug.h - Library for Arduino Wifi shield.

  Based on and modified from

  1) WiFi101 Library                 https://github.com/arduino-libraries/WiFi101

  Forked then modified by Khoi Hoang https://github.com/khoih-prog/WiFi101

  Copyright (c) 2011-2014 Arduino.  All right reserved.

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

  Version: 1.0.0

  Version Modified By   Date      Comments
  ------- -----------  ---------- -----------
  1.0.0   K Hoang      14/11/2022 Fix severe limitation to permit sending much larger data than total 4K
 ***********************************************************************************************************************************/

#pragma once

#ifndef WIFI101_DEBUG_H
#define WIFI101_DEBUG_H

#include <stdio.h>
#include <string.h>

////////////////////////////////////////

// KH Add, v1.6.0
#ifdef DEBUG_WIFININA_PORT
  #define DBG_PORT_NN      DEBUG_WIFININA_PORT
#else
  #define DBG_PORT_NN     Serial
#endif

////////////////////////////////////////

// Change _WIFININA_LOGLEVEL_ to set tracing and logging verbosity
// 0: DISABLED: no logging
// 1: ERROR: errors
// 2: WARN: errors and warnings
// 3: INFO: errors, warnings and informational (default)
// 4: DEBUG: errors, warnings, informational and debug

////////////////////////////////////////

#ifndef _WIFININA_LOGLEVEL_
  #define _WIFININA_LOGLEVEL_       0
#endif

///////////////////////////////////////

const char NN_MARK[]  = "[NN] ";
const char NN_SP[]    = " ";

#define NN_PRINT          DBG_PORT_NN.print
#define NN_PRINTLN        DBG_PORT_NN.println
#define NN_FLUSH          DBG_PORT_NN.flush

#define NN_PRINT_MARK     NN_PRINT(NN_MARK)
#define NN_PRINT_SP       NN_PRINT(NN_SP)

///////////////////////////////////////

// Error waitResponse message
#define ERROR_RESPONSE  ":Error waitResponse"

#define NN_LOGERROR0(x)     if(_WIFININA_LOGLEVEL_>0) { NN_PRINT(x); }
#define NN_LOGERROR(x)      if(_WIFININA_LOGLEVEL_>0) { NN_PRINT_MARK; NN_PRINTLN(x); }
#define NN_LOGERROR1(x,y)   if(_WIFININA_LOGLEVEL_>0) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINTLN(y); }
#define NN_LOGERROR2(x,y,z) if(_WIFININA_LOGLEVEL_>0) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINT(y); NN_PRINT_SP; NN_PRINTLN(z); }
#define NN_LOGERROR3(x,y,z,w) if(_WIFININA_LOGLEVEL_>0) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINT(y); NN_PRINT_SP; NN_PRINT(z); NN_PRINT_SP; NN_PRINTLN(w); }

///////////////////////////////////////

#define NN_LOGWARN0(x)     if(_WIFININA_LOGLEVEL_>1) { NN_PRINT(x); }
#define NN_LOGWARN(x)      if(_WIFININA_LOGLEVEL_>1) { NN_PRINT_MARK; NN_PRINTLN(x); }
#define NN_LOGWARN1(x,y)   if(_WIFININA_LOGLEVEL_>1) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINTLN(y); }
#define NN_LOGWARN2(x,y,z) if(_WIFININA_LOGLEVEL_>1) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINT(y); NN_PRINT_SP; NN_PRINTLN(z); }
#define NN_LOGWARN3(x,y,z,w) if(_WIFININA_LOGLEVEL_>1) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINT(y); NN_PRINT_SP; NN_PRINT(z); NN_PRINT_SP; NN_PRINTLN(w); }

///////////////////////////////////////

#define NN_LOGINFO0(x)     if(_WIFININA_LOGLEVEL_>2) { NN_PRINT(x); }
#define NN_LOGINFO(x)      if(_WIFININA_LOGLEVEL_>2) { NN_PRINT_MARK; NN_PRINTLN(x); }
#define NN_LOGINFO1(x,y)   if(_WIFININA_LOGLEVEL_>2) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINTLN(y); }
#define NN_LOGINFO2(x,y,z) if(_WIFININA_LOGLEVEL_>3) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINT(y); NN_PRINT_SP; NN_PRINTLN(z); }
#define NN_LOGINFO3(x,y,z,w) if(_WIFININA_LOGLEVEL_>3) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINT(y); NN_PRINT_SP; NN_PRINT(z); NN_PRINT_SP; NN_PRINTLN(w); }

///////////////////////////////////////

#define NN_LOGDEBUG0(x)     if(_WIFININA_LOGLEVEL_>3) { NN_PRINT(x); }
#define NN_LOGDEBUG(x)      if(_WIFININA_LOGLEVEL_>3) { NN_PRINT_MARK; NN_PRINTLN(x); }
#define NN_LOGDEBUG1(x,y)   if(_WIFININA_LOGLEVEL_>3) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINTLN(y); }
#define NN_LOGDEBUG2(x,y,z) if(_WIFININA_LOGLEVEL_>3) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINT(y); NN_PRINT_SP; NN_PRINTLN(z); }
#define NN_LOGDEBUG3(x,y,z,w) if(_WIFININA_LOGLEVEL_>3) { NN_PRINT_MARK; NN_PRINT(x); NN_PRINT_SP; NN_PRINT(y); NN_PRINT_SP; NN_PRINT(z); NN_PRINT_SP; NN_PRINTLN(w); }

///////////////////////////////////////

#define PRINT_FILE_LINE() do {            \
    Serial.print("[");Serial.print(__FILE__);   \
    Serial.print("::");Serial.print(__LINE__);Serial.print("]");\
} while (0);

//KH
//#define _DEBUG_

#ifdef _DEBUG_

#define INFO(format, args...) do { \
    char buf[250];  \
    sprintf(buf, format, args); \
    Serial.println(buf); \
  } while(0);

#define INFO1(x) do { PRINT_FILE_LINE() Serial.print("-I-");\
      Serial.println(x);          \
  } while (0);

#define INFO2(x,y) do { PRINT_FILE_LINE() Serial.print("-I-");\
      Serial.print(x,16);Serial.print(",");Serial.println(y,16); \
  } while (0);


#else
#define INFO1(x) do {} while(0);
#define INFO2(x,y) do {} while(0);
#define INFO(format, args...) do {} while(0);
#endif

#define WARN(args) do {} while (0);

//KH
#define _DEBUG_SPI_     false

#if _DEBUG_SPI_
#define DBG_PIN2 5
#define DBG_PIN 4

#define START()         digitalWrite(DBG_PIN2, HIGH);
#define END()           digitalWrite(DBG_PIN2, LOW);
#define SET_TRIGGER()   digitalWrite(DBG_PIN, HIGH);
#define RST_TRIGGER()   digitalWrite(DBG_PIN, LOW);

#define INIT_TRIGGER()  pinMode(DBG_PIN, OUTPUT); \
                          pinMode(DBG_PIN2, OUTPUT); \
                          RST_TRIGGER()
#define TOGGLE_TRIGGER() SET_TRIGGER() \
                             delayMicroseconds(2);    \
                                 RST_TRIGGER()
#else
#define START()
#define END()
#define SET_TRIGGER()
#define RST_TRIGGER()
#define INIT_TRIGGER()
#define TOGGLE_TRIGGER()
#endif


#endif    // WIFI101_DEBUG_H
