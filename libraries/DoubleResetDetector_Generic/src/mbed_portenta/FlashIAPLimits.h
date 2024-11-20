/**********************************************************************************************************************************
  FlashIAPLimits.h - Filesystem wrapper for LittleFS on the Mbed Portenta_H7
  
  Arduino AVR, Teensy, SAM-DUE, SAMD, STM32, nRF52, etc. boards

  DoubleResetDetector_Generic is a library for the Arduino AVR, Teensy, SAM-DUE, SAMD, STM32, nRF52, etc. boards
  to enable trigger configure mode by resetting the boards twice within configurable timeout seconds.

  Based on and modified from DataCute https://github.com/datacute/DoubleResetDetector and 
  https://github.com/khoih-prog/ESP_DoubleResetDetector 

  Built by Khoi Hoang https://github.com/khoih-prog/DoubleResetDetector_Generic
  Licensed under MIT license

  Version: 1.8.1

  Version Modified By   Date      Comments
  ------- -----------  ---------- -----------
  1.0.0   K Hoang      14/04/2020 Initial coding for boards such as AVR, Teensy, SAM DUE, SAMD and STM32, etc.
  1.0.1   K Hoang      01/05/2020 Add support to Adafruit nRF52 boards, such as Feather, Itsy-Bitsy nRF52840, NINA_W302_ublox.
  1.0.2   K Hoang      04/05/2020 Fix not-detected DRD bug for SAMD boards.
  1.0.3   K Hoang      28/12/2020 Suppress all possible compiler warnings
  1.1.0   K Hoang      27/04/2021 Use new FlashStorage_STM32 library. Add support to new STM32 core v2.0.0 and STM32L5
  1.2.0   K Hoang      12/05/2021 Add support to RASPBERRY_PI_PICO using Arduino-pico core
  1.3.0   K Hoang      28/05/2021 Add support to Nano_RP2040_Connect, RASPBERRY_PI_PICO using RP2040 Arduino mbed core
  1.4.0   K Hoang      05/06/2021 Permit more control over LittleFS for RP2040 Arduino mbed core
  1.5.0   K Hoang      07/08/2021 Add support to RTL8720DN, etc. using AmebaD core
  1.6.0   K Hoang      29/08/2021 Add support to MBED Nano_33_BLE, Nano_33_BLE_Sense, etc. using LittleFS
  1.7.0   K Hoang      10/09/2021 Add support to MBED Portenta_H7 using LittleFS
  1.7.1   K Hoang      13/09/2021 Select fix LittleFS size of 1024KB
  1.7.2   K Hoang      14/09/2021 Back to using auto LittleFS to fix bug
  1.7.3   K Hoang      10/10/2021 Update `platform.ini` and `library.json`
  1.8.0   K Hoang      26/01/2022 Update to be compatible with new FlashStorage libraries. Add support to more SAMD/STM32 boards
  1.8.1   K Hoang      05/03/2022 Add waitingForDRD() function to signal in DRD wating period
 **********************************************************************************************************************************/

#pragma once

#ifndef _FlashIAPLimits_h_
#define _FlashIAPLimits_h_

#include <Arduino.h>
#include <FlashIAP.h>
#include <FlashIAPBlockDevice.h>

using namespace mbed;

// A helper struct for FlashIAP limits
struct FlashIAPLimits 
{
  size_t flash_size;
  uint32_t start_address;
  uint32_t available_size;
};

// Get the actual start address and available size for the FlashIAP Block Device
// considering the space already occupied by the sketch (firmware).
static FlashIAPLimits getFlashIAPLimits()
{
  // Alignment lambdas
  auto align_down = [](uint64_t val, uint64_t size) 
  {
    return (((val) / size)) * size;
  };
  
  auto align_up = [](uint32_t val, uint32_t size) 
  {
    return (((val - 1) / size) + 1) * size;
  };

  FlashIAPLimits flashIAPLimits;
  
  uint32_t  flash_start_address;
  
  FlashIAP  flash;

  auto result = flash.init();
  
  if (result != 0)
    return { };

  // Find the start of first sector after text area
  int sector_size     = flash.get_sector_size(FLASHIAP_APP_ROM_END_ADDR);

  flash_start_address           = flash.get_flash_start();
  flashIAPLimits.start_address  = align_up(FLASHIAP_APP_ROM_END_ADDR, sector_size);
  flashIAPLimits.flash_size     = flash.get_flash_size();

  result = flash.deinit();

  flashIAPLimits.available_size = flash_start_address + flashIAPLimits.flash_size - flashIAPLimits.start_address;
  
  if (flashIAPLimits.available_size % (sector_size * 2)) 
  {
    flashIAPLimits.available_size = align_down(flashIAPLimits.available_size, sector_size * 2);
  }

  return flashIAPLimits;
}
#endif    // _FlashIAPLimits_h_
