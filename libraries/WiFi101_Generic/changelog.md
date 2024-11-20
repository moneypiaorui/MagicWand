## WiFi101_Generic Changelog

[![GitHub release](https://img.shields.io/github/release/khoih-prog/WiFi101_Generic.svg)](https://github.com/khoih-prog/WiFi101_Generic/releases)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/khoih-prog/WiFi101_Generic/blob/master/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)
[![GitHub issues](https://img.shields.io/github/issues/khoih-prog/WiFi101_Generic.svg)](http://github.com/khoih-prog/WiFi101_Generic/issues)

<a href="https://profile-counter.glitch.me/khoih-prog/count.svg" title="Total khoih-prog Visitor count"><img src="https://profile-counter.glitch.me/khoih-prog/count.svg" style="height: 30px;width: 200px;"></a>
<a href="https://profile-counter.glitch.me/khoih-prog-WiFi101_Generic/count.svg" title="WiFi101_Generic Visitor count"><img src="https://profile-counter.glitch.me/khoih-prog-WiFi101_Generic/count.svg" style="height: 30px;width: 200px;"></a>

---
---

## Table of Contents

* [Changelog](#changelog)
  * [Releases v1.0.0](#releases-v100)
  * [Releases v0.16.1](#releases-v0161)
  * [Releases v0.16.0](#releases-v0160)
  * [Releases v0.15.3](#releases-v0153)
  * [Releases v0.15.2](#releases-v0152)
  * [Releases v0.15.1](#releases-v0151)
  * [Releases v0.15.0](#releases-v0150)
  * [Releases v0.14.3](#releases-v0143)
  * [Releases v0.14.2](#releases-v0142)
  * [Releases v0.14.1](#releases-v0141)
  * [Releases v0.14.0](#releases-v0140)
  * [Releases v0.13.0](#releases-v0130)
  * [Releases v0.12.1](#releases-v0121)
  * [Releases v0.12.0](#releases-v0120)
  * [Releases v0.11.2](#releases-v0112)
  * [Releases v0.11.1](#releases-v0111)
  * [Releases v0.11.0](#releases-v0110)
  * [Releases v0.10.0](#releases-v0100)
  * [Releases v0.9.1](#releases-v091)
  * [Releases v0.9.0](#releases-v090)
  * [Releases v0.8.0](#releases-v080)
  * [Releases v0.7.0](#releases-v070)
  * [Releases v0.6.0](#releases-v060)
  * [Releases v0.5.1](#releases-v051)
  * [Releases v0.5.0](#releases-v050)
  
---
---

## Changelog

### Releases v1.0.0

- 2022.11.17

1. Fix severe limitation to permit sending much larger data than total 4K
2. Use `allman astyle` and add `utils`
3. Add `Packages' Patches`

### Releases v0.16.1

- 2021.05.21

* Add compatibility to [WebSockets2_Generic](https://github.com/khoih-prog/WebSockets2_Generic) library

### Releases v0.16.0

- 2019.04.04

* Added WiFi.setTimeout(timeout) API to set timeout of WiFi.begin(...)
* Changed check firmware version comparison from == to >=
* Updated latest firmware version for model B to 19.6.1

### Releases v0.15.3

- 2018.11.21

* Fixed unreachable destination issue in UDP packet transmission
* Changed where required, in library's examples, the server's URL arduino.cc\80 to example.org\80, because http://arduino.cc will be no longer available
* Changed the MAC address print in the library's example now are showed correctly all the MAC address chars
* Fixed WiFiUDP::endPacket() return value in accord with sending returns. Thanks to @mjlitke @jrowberg
* Changed WiFiSocketClass::create(...) return condition. Thanks to @jrowberg
* Changed server and URL to avoid 301 response from Server
* Fixed initialization Server issue by adding socket initialization in WiFiServer::WiFiServer constructor

### Releases v0.15.2 

- 2018.02.20

* Fixed issue with WiFiServer::begin() stopping first listening server

### Releases v0.15.1 

- 2018.02.14

* Fixed compatibility with Arduino Due
* Fixed return code of WiFiUdp::endPacket(). Thanks @mlitke
* Fixed WiFiServer::available() hanging after WiFi.end() is called

### Releases v0.15.0 

- 2018.01.02

* Fixed WiFi.status() hanging if no shield was present
* Reworked socket buffer layer to prevent (SAMD) and reduce lockups (AVR)
* Make reset pin optional. Thanks @awatterott
* Added ability to retrieve remote IP and port of WiFiClient

### Releases v0.14.3 

- 2017.06.01

* Fixed issues with WiFiMDNSResponder and large request packets
* Fixed issues with WiFiClient and sending data after the socket is closed

### Releases v0.14.2 

- 2017.05.08

* Fixed issues with WiFiServer::write not working

### Releases v0.14.1 

- 2017.04.20

* Fixed issues with WiFiMDNSResponder and Windows using Bonjour
* Correct cast of buffer size when processing received data

### Releases v0.14.0 

- 2017.03.22

* Added support for firmware 19.5.2
* Add ability to create Access Point with WPA security (f/w 19.5.2 or higher)
* Add WiFi.hostname(name) method to set custom host name for DHCP (f/w 19.5.2 or higher)
* Enables support for AES-256 Ciphers (f/w 19.5.2 or higher)
* Make provisioning mode backwards compatible with f/w 19.4.4 and older

### Releases v0.13.0 

- 2017.03.01

* Added WiFi.channel() and WiFi.BSSID()

### Releases v0.12.1 

- 2017.01.19

* Fixed tomorrow day issue in WiFi.getTime()

### Releases v0.12.0 

- 2017.01.05

* Made provisioning mode easier and added example sketch
* Fixed WiFi.getTime() not returning 0, if time has not been synced via NTP
* Fixed crashing when connecting after scanning
* Fixed WiFiServer::available() returning valid client on outbound client connection
* Added WiFiUdp::beginMulticast(port) function for compatibility with EthernetUdp, as beginMulti was inconsistent

### Releases v0.11.2 

- 2016.12.15

* Fixed value of WiFi.getTime() being off by one day
* Fixed calling WiFi.RSSI() causing lockups

### Releases v0.11.1 

- 2016.11.29

* Fixed regression for non-AVR boards that resulted in corrupt with data over than 1400 bytes was received

### Releases v0.11.0 

- 2016.11.14

* Changed WiFi.ping(...) to return round trip time on success, negative value on error. Thanks @PKGeorgiev
* WiFi.end() now powers down the WiFi module
* WiFi.config(ip) can now be used to set a static IP for WiFi.begin(...)
* Fixed WiFi.BSSID(bssid) returning reversed MAC address
* Added WiFi.APClientMacAddress() API to get MAC address of AP client in AP mode
* Added WiFi.getTime() API to get epoch from NTP

### Releases v0.10.0 

- 2016.09.08

* Added WiFi.end() to disconnect from the AP or end AP mode
* Added new WiFi.ping(...) functionality. Thanks @PKGeorgiev
* Added WiFi.setPins(...) to customize the CS, INTN, RESET and CHIPEN pins
* Add new WL_AP_LISTENING, WL_AP_CONNECTED, and WL_AP_FAILED status types for AP mode
* Fixed return value of WiFiUDP::beginPacket(host, port) when host is successfully resolved
* Added power management methods: WiFi.lowPowerMode(), WiFi.maxLowPowerMode(), WiFi.noLowPowerMode()
* Close TCP sockets when physical link is disconnected
* Fixed WiFi.RSSI() returning 0 when there was pending socket data

### Releases v0.9.1 

- 2016.04.19

* Increased compatibility with 3rd party boards and architectures.
* Rename WiFiMdnsResponder.h/cpp to WiFiMDNSResponder.h/cpp
* Fixed buffering of UDP packet

### Releases v0.9.0 

- 2016.03.21

* Fixed WiFi.status() hanging when shield is not present
* Fixed MAC address returning 00:00:00:00:00:00 with firmware version 19.3.0
* Fixed SSL write functionality with firmware version 19.3.0
* Fixed previous version of the library not working on the Uno and other boards that used PCINT's
* Added beginAP(...) API's to create access point with WEP passwords
* Fixed beginAP(...) channel off by one error
* Fixed WiFi.status() always returning WL_CONNECTED once connected, even if access point is disconnected later
* Added beginMulti API for Multicast UDP support
* Added WiFiMDNSResponder class and MDNS Web server example

### Releases v0.8.0 

- 2016.02.15

* Added example for Access Point web server (thanks @ladyada)
* Fixed MAC Address printed in reverse order
* Allow another library to override PCINT ISR (fix issues when using the WiFi101
  library with other libraries like SoftwareSerial)

### Releases v0.7.0 

- 2015.01.11

* Added support for WiFi Firmware 19.4.4
* WiFi.hostByName(...) will not try to resolve the domain name if it's already a numeric IP
* Fixed manual IP configuration (no DHCP)
* Fixed WiFiServer.available(), now follows API specification
* Fixed WEP key connection
* Fixed WiFiClient copy constructor and assignment operator. This improves stability when
  Client objects are assigned or returned from functions.
* Control pins are now configurable through defines from variant.

### Releases v0.6.0 

- 2015.11.27

* Fixed bug with AVR boards when Web Server is used
* Fixed UDP read bug on AVR Boards
* Added missing include for SSL Client
* Fixed peek() function
* Fixed some examples

### Releases v0.5.1 

- 2015.10.06

* Improved support for AVR Boards (Uno, Mega, Leonardo, etc.) and
  ARM based boards (Due and Zero).

### Releases v0.5.0 

- 2015.10.01

* Initial release


