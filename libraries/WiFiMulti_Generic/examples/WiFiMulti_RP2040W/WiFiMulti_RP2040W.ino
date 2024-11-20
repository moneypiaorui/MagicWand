/**************************************************************************************************************************************
  WiFiMulti_RP2040W.ino
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
 ***************************************************************************************************************************************/

#include "defines.h"

WiFiMulti_Generic wifiMulti;

bool isWiFiConnected()
{
  // You can change longer or shorter depending on your network response
  // Shorter => more responsive, but more ping traffic
  static uint8_t theTTL = 10;

  // Use ping() to test TCP connections
  if (WiFi.ping(WiFi.gatewayIP(), theTTL) == theTTL)
  {
    return true;
  }

  return false;
}

void heartBeatPrint()
{
  static int num = 1;

  //WFM_LOGDEBUG1("\nWiFi connected, RSSI:", WiFi.RSSI());

  if (WiFi.status() == WL_CONNECTED)
    Serial.print(F("H"));        // H means connected to WiFi
  else
    Serial.print(F("F"));        // F means not connected to WiFi

  if (num == 80)
  {
    Serial.println();
    num = 1;
  }
  else if (num++ % 10 == 0)
  {
    Serial.print(F(" "));
  } 
}

uint8_t connectMultiWiFi()
{
// For general board, this better be 1000 to enable connect the 1st time
#define WIFI_MULTI_1ST_CONNECT_WAITING_MS               1000L

#define WIFI_MULTI_CONNECT_WAITING_MS                   500L

  Serial.println("WiFi lost. Trying to scan and reconnect");

  WiFi.disconnect();

  int i = 0;

  uint8_t status = wifiMulti.run();

  delay(WIFI_MULTI_1ST_CONNECT_WAITING_MS);

  while ( ( i++ < 20 ) && ( status != WL_CONNECTED ) )
  {
    status = WiFi.status();

    if ( status == WL_CONNECTED )
      break;
    else
      delay(WIFI_MULTI_CONNECT_WAITING_MS);
  }

  if ( status == WL_CONNECTED )
  {
    WFM_LOGERROR1(F("WiFi connected after time: "), i);
    WFM_LOGERROR3(F("SSID:"), WiFi.SSID(), F(", IP address:"), WiFi.localIP());
  }
  else
  {
    WFM_LOGERROR(F("WiFi not connected"));

    if (wifiMulti.run() != WL_CONNECTED)
    {
      Serial.println("WiFi not connected!");
      delay(1000);
    }
  }

  return status;
}

void check_WiFi()
{
  //if ( (WiFi.status() != WL_CONNECTED) )
  // Klugde to temporarily fix RP2040W WiFi.status() bug ( https://github.com/earlephilhower/arduino-pico/issues/762 )
  if (!isWiFiConnected())
  {
    Serial.println(F("\nWiFi lost. Call connectMultiWiFi in loop"));
    connectMultiWiFi();
  }
}

void check_status()
{
  static uint32_t checkstatus_timeout  = 0;
  static uint32_t checkwifi_timeout    = 0;

  static uint32_t current_millis;

// You can change longer or shorter depending on your network response
// Shorter => more responsive, but more ping traffic
#define WIFICHECK_INTERVAL    1000L

#define HEARTBEAT_INTERVAL    10000L

  current_millis = millis();

  // Check WiFi every WIFICHECK_INTERVAL (10) seconds.
  if ((current_millis > checkwifi_timeout) || (checkwifi_timeout == 0))
  {
    check_WiFi();
    checkwifi_timeout = current_millis + WIFICHECK_INTERVAL;
  }

  // Print hearbeat every HEARTBEAT_INTERVAL (10) seconds.
  if ((current_millis > checkstatus_timeout) || (checkstatus_timeout == 0))
  {
    heartBeatPrint();
    checkstatus_timeout = current_millis + HEARTBEAT_INTERVAL;
  }

  // Important delay() for RTL8720DN
  delay(200);
}

void setup()
{
  Serial.begin(115200);
  while (!Serial && millis() < 5000);

  Serial.print(F("\nStarting WiFiMulti_RP2040W on ")); Serial.println(BOARD_NAME);
  Serial.println(WIFIMULTI_GENERIC_VERSION);

  wifiMulti.addAP(your_ssid,  your_pass);
  wifiMulti.addAP(your_ssid1, your_pass1);
  wifiMulti.addAP("ssid_from_AP_1", "your_password_for_AP_1");
  wifiMulti.addAP("ssid_from_AP_2", "your_password_for_AP_2");
  wifiMulti.addAP("ssid_from_AP_3", "your_password_for_AP_3");

  Serial.println("Connecting WiFi...");

  if (wifiMulti.run() == WL_CONNECTED)
  {
    Serial.print("\nWiFi connected, SSID: ");
    Serial.print(WiFi.SSID());
    Serial.print(", IP address: ");
    Serial.println(WiFi.localIP());
  }
}

void loop()
{
  check_status();
}
