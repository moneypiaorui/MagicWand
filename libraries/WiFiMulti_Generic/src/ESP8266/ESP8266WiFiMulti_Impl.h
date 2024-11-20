/**************************************************************************************************************************************
  ESP8266WiFiMulti_Impl.h
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
  This file is part of the esp32 core for Arduino environment.
  
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

#ifndef _ESP8266_WIFIMULTI_GENERIC_IMPL_H_
#define _ESP8266_WIFIMULTI_GENERIC_IMPL_H_  

#include "PolledTimeout.h"
#include "WiFiMulti_Generic.h"
#include <limits.h>
#include <string.h>


/**
   @brief Print WiFi status
   @details
        Macro DEBUG_ESP_WIFI and DEBUG_ESP_PORT must be configured
   @param status
        WiFi status
*/
static void printWiFiStatus(wl_status_t status)
{
#ifdef DEBUG_ESP_WIFI
  IPAddress ip;
  uint8_t *mac;

  switch (status) 
  {
    case WL_CONNECTED:
      ip = WiFi.localIP();
      mac = WiFi.BSSID();

      WFM_LOGDEBUG("Connected:");
      WFM_LOGDEBUG3("SSID:", WiFi.SSID().c_str(), ", IP:", ip);
      WFM_LOGDEBUG5_SEPARATOR(":", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
      WFM_LOGDEBUG3("Channel:", WiFi.channel(), ", RSSI:", WiFi.RSSI());

      break;
    case WL_NO_SSID_AVAIL:
      WFM_LOGDEBUG("Connecting failed AP not found.");
      break;
    case WL_CONNECT_FAILED:
      WFM_LOGDEBUG("Connecting failed.");
      break;
    case WL_WRONG_PASSWORD:
      WFM_LOGDEBUG("Wrong password.");
      break;
    default:
      WFM_LOGDEBUG1("Connecting failed, status =", status);
      break;
  }
#else
  // Suppress warning unused variable
  (void)(status);
#endif
}

/**
   @brief Wait for WiFi connect status change, protected with timeout
   @param connectTimeoutMs
        WiFi connection timeout in ms
   @return
        WiFi connection status
*/
static wl_status_t waitWiFiConnect(const uint32_t& connectTimeoutMs)
{
  wl_status_t status;

  // Set WiFi connect timeout
  using esp8266::polledTimeout::oneShotMs;
  oneShotMs connectTimeout(connectTimeoutMs);

  // Wait for WiFi status change or timeout
  do 
  {
    // Refresh watchdog
    delay(0);

    // Get WiFi status
    status = WiFi.status();

    // Check status
    if (status == WL_CONNECTED) 
    {
      // Connected, print WiFi status
      printWiFiStatus(status);

      // Return WiFi status
      return status;
    } 
    else if (status == WL_CONNECT_FAILED) 
    {
      WFM_LOGDEBUG("Connect failed");

      // Return WiFi connect failed
      return WL_CONNECT_FAILED;
    }
  } while (!connectTimeout);

  WFM_LOGDEBUG("Connect timeout");

  return WL_CONNECT_FAILED;
}

/**
   @brief Constructor
*/
WiFiMulti_Generic::WiFiMulti_Generic() : _firstRun(true)
{
}

/**
   @brief Destructor
*/
WiFiMulti_Generic::~WiFiMulti_Generic()
{
  // Cleanup memory
  APlistClean();
}

/**
   @brief Add Access Point
   @param ssid
        WiFi SSID char array, max 32 characters + NULL character
   @param passphrase
        WiFi password char array, max 63 characters + NULL character
   @retval true
        Success
   @retval false
        Failure
*/
bool WiFiMulti_Generic::addAP(const char *ssid, const char *passphrase)
{
  return APlistAdd(ssid, passphrase);
}

/**
   @brief Remove all Access Points from list
*/
void WiFiMulti_Generic::cleanAPlist()
{
  APlistClean();
}

/**
   @brief Check if Access Point exists in list
   @param ssid
        WiFi SSID
   @param passphrase
        WiFi Password
   @retval true
        Success
   @retval false
        Failure
*/
bool WiFiMulti_Generic::existsAP(const char *ssid, const char *passphrase)
{
  return APlistExists(ssid, passphrase);
}

/**
   @brief Keep WiFi connected to Access Point with strongest WiFi signal (RSSI)
   @param connectTimeoutMs
        Timeout in ms per WiFi connection (excluding fixed 5 seconds scan timeout)
   @return
        WiFi status
*/
wl_status_t WiFiMulti_Generic::run(const uint32_t& connectTimeoutMs)
{
  int8_t scanResult;
  wl_status_t status;

  // Fast connect to previous WiFi on startup
  if (_firstRun) 
  {
    _firstRun = false;

    // Check if previous WiFi connection saved
    if (strlen(WiFi.SSID().c_str())) 
    {
      WFM_LOGDEBUG("Connecting saved WiFi");

      // Connect to previous saved WiFi
      WiFi.begin();

      // Wait for status change
      status = waitWiFiConnect(connectTimeoutMs);
    }
  }

  // Check connection state
  status = WiFi.status();
  
  if (status == WL_CONNECTED) 
  {
    // Already connected
    return status;
  }

  // Start WiFi scan
  scanResult = startScan();
  
  if (scanResult < 0) 
  {
    // No WiFi scan results
    return WL_NO_SSID_AVAIL;
  }

  // Try to connect to multiple WiFi's with strongest signal (RSSI)
  return connectWiFiMulti(connectTimeoutMs);
}

/**
   @brief Start WiFi scan
   @retval >0
        Number of detected WiFi SSID's
   @retval 0
        No WiFi connections found
   @retval -2
        WiFi scan failed
*/
int8_t WiFiMulti_Generic::startScan()
{
  int8_t scanResult;

  WFM_LOGDEBUG("Start scan");

  // Clean previous scan
  WiFi.scanDelete();

  // Remove previous WiFi SSID/password
  WiFi.disconnect();

  // Start wifi scan in async mode
  WiFi.scanNetworks(true);

  // Set WiFi scan timeout
  using esp8266::polledTimeout::oneShotMs;
  oneShotMs scanTimeout(WIFI_SCAN_TIMEOUT_MS);

  // Wait for WiFi scan change or timeout
  do 
  {
    // Refresh watchdog
    delay(0);

    // Check scan timeout which may occur when scan does not report completion
    if (scanTimeout) 
    {
      WFM_LOGDEBUG("Scan timeout");
      return WIFI_SCAN_FAILED;
    }

    // Get scan result
    scanResult = WiFi.scanComplete();
  } while (scanResult < 0);

  // Print WiFi scan result
  printWiFiScan();

  // Return (positive) number of detected WiFi networks
  return scanResult;
}

/**
   @brief Connect to multiple WiFi's
   @param connectTimeoutMs
        WiFi connect timeout in ms
   @return
        WiFi connection status
*/
wl_status_t WiFiMulti_Generic::connectWiFiMulti(const uint32_t& connectTimeoutMs)
{
  int8_t scanResult;
  String ssid;
  int32_t rssi;
  uint8_t encType;
  uint8_t *bssid;
  int32_t channel;
  bool hidden;

  // Get scan results
  scanResult = WiFi.scanComplete();

  // Find known WiFi networks
  uint8_t known[_APlist.size()];
  uint8_t numNetworks = 0;
  
  for (int8_t i = 0; i < scanResult; i++) 
  {
    // Get network information
    WiFi.getNetworkInfo(i, ssid, encType, rssi, bssid, channel, hidden);

    // Check if the WiFi network contains an entry in AP list
    for (auto entry : _APlist) 
    {
      // Check SSID
      if (ssid == entry.ssid) 
      {
        // Known network
        known[numNetworks++] = i;
      }
    }
  }

  // Sort WiFi networks by RSSI
  for (int i = 0; i < numNetworks; i++) 
  {
    for (int j = i + 1; j < numNetworks; j++) 
    {
      if (WiFi.RSSI(known[j]) > WiFi.RSSI(known[i])) 
      {
        int8_t tmp;

        // Swap indices
        tmp = known[i];
        known[i] = known[j];
        known[j] = tmp;
      }
    }
  }

  // Print sorted indices
  WFM_LOGDEBUG("Sorted indices: ");
  
  for (int8_t i = 0; i < numNetworks; i++) 
  {
    WFM_LOGDEBUG1("Known networks index =", known[i]);
  }
  
  //WFM_LOGDEBUG0("\n");

  // Create indices for AP connection failures
  uint8_t connectSkipIndex[_APlist.size()];
  memset(connectSkipIndex, 0, sizeof(connectSkipIndex));

  // Connect to known WiFi AP's sorted by RSSI
  for (int8_t i = 0; i < numNetworks; i++) 
  {
    // Get network information
    WiFi.getNetworkInfo(known[i], ssid, encType, rssi, bssid, channel, hidden);

    for (uint8_t j = 0; j < _APlist.size(); j++) 
    {
      auto &entry = _APlist[j];

      // Check SSID
      if (ssid == entry.ssid) 
      {
        WFM_LOGDEBUG1("Connecting SSID =", ssid);

        // Connect to WiFi
        WiFi.begin(ssid, entry.passphrase, channel, bssid);

        // Wait for status change
        if (waitWiFiConnect(connectTimeoutMs) == WL_CONNECTED) 
        {
          return WL_CONNECTED;
        }

        // Failed to connect, skip for hidden SSID connects
        connectSkipIndex[j] = true;
      }
    }
  }

  // Try to connect to hidden AP's which are not reported by WiFi scan
  for (uint8_t i = 0; i < _APlist.size(); i++) 
  {
    auto &entry = _APlist[i];

    if (!connectSkipIndex[i]) 
    {
      WFM_LOGDEBUG1("Try hidden connect SSID =", entry.ssid);

      // Connect to WiFi
      WiFi.begin(entry.ssid, entry.passphrase);

      // Wait for status change
      if (waitWiFiConnect(connectTimeoutMs) == WL_CONNECTED) 
      {
        return WL_CONNECTED;
      }
    }
  }

  WFM_LOGDEBUG1("Could not connect, SSID =", ssid);

  // Could not connect to any WiFi network
  return WL_CONNECT_FAILED;
}

// ##################################################################################

/**
   @brief Add WiFi connection to internal AP list
   @param ssid
        WiFi SSID
   @param passphrase
        WiFi Password
   @retval true
        Success
   @retval false
        Failure
*/
bool WiFiMulti_Generic::APlistAdd(const char *ssid, const char *passphrase)
{
  WifiAPEntry newAP;

  if (!ssid || (*ssid == 0x00) || (strlen(ssid) > 31)) 
  {
    // Fail SSID too long or missing!
    WFM_LOGDEBUG("[APlistAdd] No ssid or ssid too long");
    return false;
  }

  // For passphrase, max is 63 ascii + null. For psk, 64hex + null.
  if (passphrase && (strlen(passphrase) > 64)) 
  {
    // fail passphrase too long!
    WFM_LOGDEBUG("[APlistAdd] Passphrase too long");
    return false;
  }

  if (APlistExists(ssid, passphrase)) 
  {
    WFM_LOGDEBUG1("[APlistAdd] SSID: Already exists, SSID =", ssid);
    return true;
  }

  newAP.ssid = strdup(ssid);

  if (!newAP.ssid) 
  {
    WFM_LOGDEBUG("[APlistAdd] Fail newAP.ssid == 0");
    return false;
  }

  if (passphrase) 
  {
    newAP.passphrase = strdup(passphrase);
  } 
  else 
  {
    newAP.passphrase = strdup("");
  }

  if (!newAP.passphrase) 
  {
    WFM_LOGDEBUG("[APlistAdd] Fail newAP.passphrase == 0");
    free(newAP.ssid);
    
    return false;
  }

  _APlist.push_back(newAP);
  
  WFM_LOGDEBUG1("[APlistAdd] Add SSID: SSID =", newAP.ssid);
  
  return true;
}

/**
   @brief Check if AP exists in list
   @param ssid
        WiFi SSID
   @param passphrase
        WiFi Password
   @return
*/
bool WiFiMulti_Generic::APlistExists(const char *ssid, const char *passphrase)
{
  if (!ssid || (*ssid == 0x00) || (strlen(ssid) > 31)) 
  {
    // Fail SSID too long or missing
    WFM_LOGDEBUG("[APlistExists] No ssid or ssid too long");
    return false;
  }

  for (auto entry : _APlist) 
  {
    if (!strcmp(entry.ssid, ssid)) 
    {
      if (!passphrase) {
        if (!strcmp(entry.passphrase, "")) 
        {
          return true;
        }
      } 
      else 
      {
        if (!strcmp(entry.passphrase, passphrase))
        {
          return true;
        }
      }
    }
  }
  
  return false;
}

/**
   @brief Remove all AP's from list
*/
void WiFiMulti_Generic::APlistClean()
{
  // Remove all entries from APlist
  for (auto entry : _APlist) 
  {
    if (entry.ssid) 
    {
      free(entry.ssid);
    }
    
    if (entry.passphrase) 
    {
      free(entry.passphrase);
    }
  }

  _APlist.clear();
}

/**
   @brief Print WiFi scan results
   @details
        Macro DEBUG_ESP_WIFI and DEBUG_ESP_PORT must be configured
*/
void WiFiMulti_Generic::printWiFiScan()
{
#ifdef DEBUG_ESP_WIFI
  String ssid;
  int32_t rssi;
  uint8_t encryptionType;
  uint8_t* bssid;
  int32_t channel;
  bool hidden;
  int8_t scanResult;

  scanResult = WiFi.scanComplete();

  WFM_LOGDEBUG1("Number of networks found =", scanResult);

  // Print unsorted scan results
  for (int8_t i = 0; i < scanResult; i++) 
  {
    bool known = false;

    WiFi.getNetworkInfo(i, ssid, encryptionType, rssi, bssid, channel, hidden);

    for (auto entry : _APlist) 
    {
      if (ssid == entry.ssid) 
      {
        // SSID match
        known = true;
      }
    }

    if (known) 
    {
      WFM_LOGDEBUG3("  Known => #", i, ", Channel:", channel);
    } 
    else 
    {
      WFM_LOGDEBUG3("Unknown => #", i, ", Channel:", chan_scan);
    }

      WFM_HEXLOGDEBUG5_SEPARATOR(":", bssid[0], bssid[1], bssid[2], bssid[3], bssid[4], bssid[5]);
      WFM_LOGDEBUG5("SSID:", ssid.c_str(), ", RSSI:", rssi, ", Secured:", (encryptionType == ENC_TYPE_NONE) ? 'n' : 'y');
    delay(0);
  }
#endif
}

#endif    // _ESP8266_WIFIMULTI_GENERIC_IMPL_H_
