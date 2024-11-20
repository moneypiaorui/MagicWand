/**********************************************************************************************************************************
  WiFiServer.cpp - Library for Arduino Wifi shield.

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

#include "utility/WiFiSocket.h"

#include "WiFi101.h"
#include "WiFiClient.h"
#include "WiFiServer.h"

////////////////////////////////////////

WiFiServer::WiFiServer(uint16_t port) :
  _socket(-1)
{
  _port = port;
}

////////////////////////////////////////

void WiFiServer::begin()
{
  begin((uint8_t) 0);
}

////////////////////////////////////////

uint8_t WiFiServer::beginSSL()
{
  return begin((uint8_t) SOCKET_FLAGS_SSL);
}

////////////////////////////////////////

// For compatibility to WebSockets2_Generic library

void WiFiServer::begin(uint16_t port)
{
  _port = port;
  begin((uint8_t) 0);
}

////////////////////////////////////////

uint8_t WiFiServer::beginSSL(uint16_t port)
{
  _port = port;
  return begin((uint8_t) SOCKET_FLAGS_SSL);
}

////////////////////////////////////////

uint8_t WiFiServer::begin(uint8_t opt)
{
  struct sockaddr_in addr;

  // Initialize socket address structure.
  addr.sin_family = AF_INET;
  addr.sin_port = _htons(_port);
  addr.sin_addr.s_addr = 0;

  if (_socket != -1 && WiFiSocket.listening(_socket))
  {
    WiFiSocket.close(_socket);
    _socket = -1;
  }

  // Open TCP server socket.
  if ((_socket = WiFiSocket.create(AF_INET, SOCK_STREAM, opt)) < 0)
  {
    return 0;
  }

  // Bind socket:
  if (!WiFiSocket.bind(_socket, (struct sockaddr *)&addr, sizeof(struct sockaddr_in)))
  {
    WiFiSocket.close(_socket);
    _socket = -1;
    return 0;
  }

  // Listen socket:
  if (!WiFiSocket.listen(_socket, 0))
  {
    WiFiSocket.close(_socket);
    _socket = -1;
    return 0;
  }

  return 1;
}

////////////////////////////////////////

WiFiClient WiFiServer::available(uint8_t* status)
{
  if (status != NULL)
  {
    *status = 0;
  }

  if (_socket != -1 && !WiFiSocket.listening(_socket))
  {
    _socket = -1;
  }

  if (_socket != -1)
  {
    SOCKET child = WiFiSocket.accepted(_socket);

    if (child > -1)
    {
      return WiFiClient(child);
    }

    for (SOCKET s = 0; s < TCP_SOCK_MAX; s++)
    {
      if (WiFiSocket.hasParent(_socket, s) && WiFiSocket.available(s))
      {
        return WiFiClient(s);
      }
    }
  }

  return WiFiClient();
}

////////////////////////////////////////

uint8_t WiFiServer::status()
{
  // Deprecated.
  return 0;
}

////////////////////////////////////////

size_t WiFiServer::write(uint8_t b)
{
  return write(&b, 1);
}

// KH rewrite to enable chunk-sending for large file

////////////////////////////////////////

// Private function
size_t WiFiServer::_write(const int sockindex, const uint8_t *buf, size_t size)
{
  int written = 0;
  int retry = WIFI_SERVER_MAX_WRITE_RETRY;

  size_t totalBytesSent = 0;
  size_t bytesRemaining = size;

  NN_LOGINFO3("WiFiServer::write: size = ", size, ", MAX_SIZE =", WIFI_SERVER_SEND_MAX_SIZE);

  if (size == 0)
  {
    setWriteError();

    NN_LOGDEBUG("WiFiServer::write: size = 0");

    return 0;
  }

  while (retry)
  {
    written =  WiFiSocket.write(sockindex, buf, min(bytesRemaining, (size_t) WIFI_SERVER_SEND_MAX_SIZE) );

    if (written > 0)
    {
      totalBytesSent += written;

      NN_LOGINFO3("WiFiServer::write: written = ", written, ", totalBytesSent =", totalBytesSent);

      if (totalBytesSent >= size)
      {
        NN_LOGINFO3("WiFiServer::write: Done, written = ", written, ", totalBytesSent =", totalBytesSent);

        //completed successfully
        retry = 0;
      }
      else
      {
        buf += written;
        bytesRemaining -= written;
        retry = WIFI_SERVER_MAX_WRITE_RETRY;

        NN_LOGINFO3("WiFiServer::write: Partially Done, written = ", written, ", bytesRemaining =", bytesRemaining);
      }
    }
    else if (written <= 0)
    {
      NN_LOGERROR("WiFiServer::write: written error");

      setWriteError();

      written = 0;
      retry = 0;
    }

    // Looping
  }

  setWriteError();

  return 0;
}

////////////////////////////////////////

size_t WiFiServer::write(const uint8_t *buffer, size_t size)
{
  if (_socket == -1)
  {
    return 0;
  }

  size_t totalBytesSent = 0;

  for (int sock = 0; sock < TCP_SOCK_MAX; sock++)
  {
    if (WiFiSocket.hasParent(_socket, sock))
    {
      totalBytesSent += _write(sock, buffer, size);
    }
  }

  return totalBytesSent;
}

////////////////////////////////////////

