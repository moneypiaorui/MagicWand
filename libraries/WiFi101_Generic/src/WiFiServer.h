/**********************************************************************************************************************************
  WiFiServer.h - Library for Arduino Wifi shield.

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

#ifndef WIFISERVER_H
#define WIFISERVER_H

#include <Arduino.h>
#include <Server.h>

////////////////////////////////////////

class WiFiClient;

////////////////////////////////////////

class WiFiServer : public Server
{
  private:
    SOCKET _socket;
    uint16_t _port;
    uint8_t begin(uint8_t opt);

    // KH
    size_t _write(const int sockindex, const uint8_t *buf, size_t size);
    //////

  public:
    WiFiServer(uint16_t);
    WiFiClient available(uint8_t* status = NULL);
    void begin();
    uint8_t beginSSL();
    // For compatibility to WebSockets2_Generic library
    void begin(uint16_t port);
    uint8_t beginSSL(uint16_t port);
    //////
    virtual size_t write(uint8_t);
    virtual size_t write(const uint8_t *buf, size_t size);
    uint8_t status();

    using Print::write;
};

////////////////////////////////////////

#endif /* WIFISERVER_H */
