/**************************************************************************************************************************************
  ESP_AT_WiFiMulti_Generic_Impl.h
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

#ifndef _ESP_AT_WIFIMULTI_GENERIC_IMPL_H_
#define _ESP_AT_WIFIMULTI_GENERIC_IMPL_H_ 

#include <limits.h>
#include <string.h>

WiFiMulti_Generic::WiFiMulti_Generic()
{
}

WiFiMulti_Generic::~WiFiMulti_Generic()
{
  for (uint32_t i = 0; i < APlist.size(); i++) 
  {
    WifiAPlist_t entry = APlist[i];
    
    if (entry.ssid) 
    {
      free(entry.ssid);
    }
    
    if (entry.passphrase) 
    {
      free(entry.passphrase);
    }
  }
  
  APlist.clear();
}

bool WiFiMulti_Generic::addAP(const char* ssid, const char *passphrase)
{
  WifiAPlist_t newAP;

  if (!ssid || *ssid == 0x00 || strlen(ssid) > 31) 
  {
    // fail SSID too long or missing!
    WFM_LOGERROR("[addAP] No ssid or ssid too long");
    return false;
  }

  if (passphrase && strlen(passphrase) > 64) 
  {
    // fail passphrase too long!
    WFM_LOGERROR("[addAP] Passphrase too long");
    return false;
  }

  newAP.ssid = strdup(ssid);

  if (!newAP.ssid) 
  {
    WFM_LOGERROR("[addAP] Fail : newAP.ssid == 0");
    return false;
  }

  if (passphrase && *passphrase != 0x00) 
  {
    newAP.passphrase = strdup(passphrase);
    
    if (!newAP.passphrase) 
    {
      WFM_LOGERROR("[addAP] Fail : newAP.passphrase == 0");
      free(newAP.ssid);
      return false;
    }
  } 
  else 
  {
    newAP.passphrase = NULL;
  }

  APlist.push_back(newAP);
  WFM_LOGINFO1("[addAP] add SSID:", newAP.ssid);
  
  return true;
}

uint8_t WiFiMulti_Generic::run(const uint32_t& connectTimeout)
{
  int8_t scanResult;
  uint8_t status = WiFi.status();
  
  if (status == WL_CONNECTED)
  {
    for (uint32_t x = 0; x < APlist.size(); x++) 
    {
      if (WiFi.SSID() == APlist[x].ssid) 
      {
        WFM_LOGINFO3("SSID =", WiFi.SSID(), "APlist.SSID =", APlist[x].ssid);
        return status;
      }
    }

    WiFi.disconnect();
   
    delay(10);
    status = WiFi.status();
  }

  scanResult = WiFi.scanNetworks();
  
  if (scanResult == 0) 
  {
    // scan is running
    return 0;
  } 
  else if (scanResult >= 0) 
  {
    // scan done analyze
    WifiAPlist_t bestNetwork { NULL, NULL };
    
    int bestNetworkDb = INT_MIN;
    uint8_t bestBSSID[6] = { 0 };
    int32_t bestChannel = 0;

    WFM_LOGINFO("[run] Scan done");

    if (scanResult == 0) 
    {
      WFM_LOGERROR("[run] No networks found");
    } 
    else 
    {
      WFM_LOGINFO1("[run] Number of Networks found:", scanResult);
      
      for (int8_t i = 0; i < scanResult; ++i) 
      {        
        String ssid_scan    = WiFi.SSID(i);
        int32_t rssi_scan   = WiFi.RSSI(i); 
        uint8_t sec_scan    = WiFi.encryptionType(i); 
        
        byte bssid[6];
        uint8_t* BSSID_scan  = WiFi.BSSID(i, bssid);
        
        int32_t chan_scan   = WiFi.channel(i);      

        bool known = false;
        
        for (uint32_t x = APlist.size() ; x > 0; x--) 
        {
          WifiAPlist_t entry = APlist[x - 1];

          if (ssid_scan == entry.ssid) 
          { 
            // SSID match
            known = true;
            
            if (rssi_scan > bestNetworkDb) 
            { 
              // best network
              if (sec_scan == ENC_TYPE_NONE || entry.passphrase) 
              { 
                // check for passphrase if not open wlan
                bestNetworkDb = rssi_scan;
                bestChannel = chan_scan;
                memcpy((void*) &bestNetwork, (void*) &entry, sizeof(bestNetwork));
                memcpy((void*) &bestBSSID, (void*) BSSID_scan, sizeof(bestBSSID));
              }
            }
            
            break;
          }
        }
        
        WFM_PRINT_LINE;

        if (known) 
        {
          WFM_LOGDEBUG3("  Known => #", i, ", Channel:", chan_scan);
          WFM_HEXLOGDEBUG5_SEPARATOR(":", BSSID_scan[0], BSSID_scan[1], BSSID_scan[2], BSSID_scan[3], BSSID_scan[4], BSSID_scan[5]);
          WFM_LOGDEBUG5("SSID:", ssid_scan.c_str(), ", RSSI:", rssi_scan, ", Secured:", (sec_scan == ENC_TYPE_NONE /*WIFI_AUTH_OPEN*/) ? 'n' : 'y');
        } 
        else 
        {
          WFM_LOGDEBUG3("Unknown => #", i, ", Channel:", chan_scan);
          WFM_HEXLOGDEBUG5_SEPARATOR(":", BSSID_scan[0], BSSID_scan[1], BSSID_scan[2], BSSID_scan[3], BSSID_scan[4], BSSID_scan[5]);
          WFM_LOGDEBUG5("SSID:", ssid_scan.c_str(), ", RSSI:", rssi_scan, ", Secured:", (sec_scan == ENC_TYPE_NONE /*WIFI_AUTH_OPEN*/) ? 'n' : 'y');
        }
      }
    }
    
    WFM_PRINT_LINE;

    if (bestNetwork.ssid) 
    {
      WFM_LOGINFO0("[run] Connecting BSSID: ");
      WFM_HEXLOGINFO5_SEPARATOR(":", bestBSSID[0], bestBSSID[1], bestBSSID[2], bestBSSID[3], bestBSSID[4], bestBSSID[5]);
      WFM_LOGINFO5("SSID: ", bestNetwork.ssid, ", Channel: ", bestChannel, ", Best dB: ", bestNetworkDb);
      
      WiFi.begin(bestNetwork.ssid, bestNetwork.passphrase);
      
      status = WiFi.status();

      auto startTime = millis();
      
      // wait for connection, fail, or timeout
      while (status != WL_CONNECTED && status != WL_CONNECT_FAILED && status != WL_CONNECT_FAILED && (millis() - startTime) <= connectTimeout) 
      {
        delay(10);
        status = WiFi.status();
      }

      switch (status) 
      {
        case WL_CONNECTED:
          WFM_LOGINFO("[run] Connecting done.");
          WFM_LOGDEBUG3("[run] SSID: ", WiFi.SSID(), ", IP: ", WiFi.localIP());
          
          break;
          
        case WL_CONNECT_FAILED:
          WFM_LOGERROR("[run] Connecting Failed.");
          break;
          
        default:
          WFM_LOGERROR1("[run] Connecting Failed, status =", status);
          break;
      }
    } 
    else 
    {
      WFM_LOGERROR("[run] no matching wifi found!");
    }
  } 
  else 
  {
    // start scan
    WFM_LOGDEBUG("[run] delete old wifi config...");
    WiFi.disconnect();

    WFM_LOGDEBUG("[run] start scan");

    WiFi.scanNetworks();
  }

  return status;
}

#endif    // _ESP_AT_WIFIMULTI_GENERIC_IMPL_H_
