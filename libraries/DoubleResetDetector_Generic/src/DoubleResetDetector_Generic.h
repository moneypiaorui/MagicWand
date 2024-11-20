/**********************************************************************************************************************************
  DoubleResetDetector_Generic.h
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

#ifndef DoubleResetDetector_Generic_H
#define DoubleResetDetector_Generic_H

#ifndef DRD_GENERIC_DEBUG
  #define DRD_GENERIC_DEBUG       false
#endif

#ifndef DOUBLERESETDETECTOR_GENERIC_VERSION
  #define DOUBLERESETDETECTOR_GENERIC_VERSION            "DoubleResetDetector_Generic v1.8.1"

  #define DOUBLERESETDETECTOR_GENERIC_VERSION_MAJOR      1
  #define DOUBLERESETDETECTOR_GENERIC_VERSION_MINOR      8
  #define DOUBLERESETDETECTOR_GENERIC_VERSION_PATCH      1

#define DOUBLERESETDETECTOR_GENERIC_VERSION_INT        1008001

#endif

#if ( defined(ESP32) || defined(ESP8266) )
  #error Please use ESP_DoubleResetDetector library (github.com/khoih-prog/ESP_DoubleResetDetector) for ESP8266 and ESP32!
#endif

// For AVR, Teensy, STM32 boards, use EEPROM
// For SAM DUE, use DueFlashStorage. For SAMD, use FlashStorage_SAMD
// For RTL8720, use FlashStorage_RTL8720

#define  DRD_FILENAME     "/drd.dat"
#define  DRD_FLAG_OFFSET  0

#if defined(ARDUINO) && (ARDUINO >= 100)
  #include <Arduino.h>
#else
  #include <WProgram.h>
#endif

#define DRD_GENERIC_USE_EEPROM      true

///////////////////////////// 

#define DRD_GENERIC_USE_SAM_DUE     false
#define DRD_GENERIC_USE_SAMD        false
#define DRD_GENERIC_USE_STM32       false
#define DRD_GENERIC_USE_NRF52       false
#define DRD_GENERIC_USE_RP2040      false
#define DRD_GENERIC_USE_MBED_RP2040 false
#define DRD_GENERIC_USE_NANO33BLE   false
#define DRD_GENERIC_USE_RTL8720     false
#define DRD_GENERIC_USE_PORTENTA    false

///////////////////////////// 

#if ( defined(ARDUINO_SAM_DUE) || defined(__SAM3X8E__) )
  #if defined(DRD_GENERIC_USE_SAM_DUE)
    #undef DRD_GENERIC_USE_SAM_DUE
  #endif
  #define DRD_GENERIC_USE_SAM_DUE     true
  #if defined(DRD_GENERIC_USE_EEPROM)
    #undef DRD_GENERIC_USE_EEPROM
  #endif
  #define DRD_GENERIC_USE_EEPROM    false
  
  #if (DRD_GENERIC_DEBUG)
    #warning Use SAM-DUE and DueFlashStorage
  #endif
  
/////////////////////////////   
#elif ( defined(ARDUINO_SAMD_ZERO) || defined(ARDUINO_SAMD_MKR1000) || defined(ARDUINO_SAMD_MKRWIFI1010) \
      || defined(ARDUINO_SAMD_NANO_33_IOT) || defined(ARDUINO_SAMD_MKRFox1200) || defined(ARDUINO_SAMD_MKRWAN1300) || defined(ARDUINO_SAMD_MKRWAN1310) \
      || defined(ARDUINO_SAMD_MKRGSM1400) || defined(ARDUINO_SAMD_MKRNB1500) || defined(ARDUINO_SAMD_MKRVIDOR4000) \
      || defined(ARDUINO_SAMD_CIRCUITPLAYGROUND_EXPRESS) || defined(__SAMD51__) || defined(__SAMD51J20A__) \
      || defined(__SAMD51J19A__) || defined(__SAMD51G19A__) || defined(__SAMD51P19A__)  \
      || defined(__SAMD21E15A__) || defined(__SAMD21E16A__) || defined(__SAMD21E17A__) || defined(__SAMD21E18A__) \
      || defined(__SAMD21G15A__) || defined(__SAMD21G16A__) || defined(__SAMD21G17A__) || defined(__SAMD21G18A__) \
      || defined(__SAMD21J15A__) || defined(__SAMD21J16A__) || defined(__SAMD21J17A__) || defined(__SAMD21J18A__) )
  #if defined(DRD_GENERIC_USE_SAMD)
    #undef DRD_GENERIC_USE_SAMD
  #endif
  #define DRD_GENERIC_USE_SAMD      true
  #if defined(DRD_GENERIC_USE_EEPROM)
    #undef DRD_GENERIC_USE_EEPROM
  #endif
  #define DRD_GENERIC_USE_EEPROM    false
  
  #if (DRD_GENERIC_DEBUG)
    #warning Use SAMD and FlashStorage
  #endif

/////////////////////////////   
#elif ( defined(NRF52840_FEATHER) || defined(NRF52832_FEATHER) || defined(NRF52_SERIES) || defined(ARDUINO_NRF52_ADAFRUIT) || \
        defined(NRF52840_FEATHER_SENSE) || defined(NRF52840_ITSYBITSY) || defined(NRF52840_CIRCUITPLAY) || defined(NRF52840_CLUE) || \
        defined(NRF52840_METRO) || defined(NRF52840_PCA10056) || defined(PARTICLE_XENON) | defined(NINA_B302_ublox) ) && \
        ! ( defined(ARDUINO_ARCH_MBED) )

  #if defined(DRD_GENERIC_USE_NRF52)
    #undef DRD_GENERIC_USE_NRF52
  #endif
  #define DRD_GENERIC_USE_NRF52      true
  #if defined(DRD_GENERIC_USE_EEPROM)
    #undef DRD_GENERIC_USE_EEPROM
  #endif
  #define DRD_GENERIC_USE_EEPROM    false
  
  #if (DRD_GENERIC_DEBUG)
    #warning Use NRF52 and LittleFS / InternalFS
  #endif
  
///////////////////////////// 
#elif ( defined(ARDUINO_ARCH_RP2040) && !defined(ARDUINO_ARCH_MBED) )

  #if defined(DRD_GENERIC_USE_RP2040)
    #undef DRD_GENERIC_USE_RP2040
  #endif
  #define DRD_GENERIC_USE_RP2040      true
  
  #if defined(DRD_GENERIC_USE_EEPROM)
    #undef DRD_GENERIC_USE_EEPROM
  #endif
  #define DRD_GENERIC_USE_EEPROM    false
  
  #if (DRD_GENERIC_DEBUG)
    #warning Use RP2040 (such as RASPBERRY_PI_PICO) and LittleFS
  #endif
  
///////////////////////////// 
#elif ( defined(ARDUINO_ARCH_RP2040) && defined(ARDUINO_ARCH_MBED) )

  // For Arduino' arduino-mbed core
  // To check and determine if we need to init LittleFS here
  #if MBED_RP2040_INITIALIZED
    #define DRD_MBED_LITTLEFS_NEED_INIT     false
    
    #if (DRD_GENERIC_DEBUG)
      #warning MBED_RP2040_INITIALIZED in another place
    #endif
    
  #else
    // Better to delay until init done
    #if defined(MBED_RP2040_INITIALIZED)
      #undef MBED_RP2040_INITIALIZED
    #endif
    #define MBED_RP2040_INITIALIZED           true
    
    #define DRD_MBED_LITTLEFS_NEED_INIT     true
  #endif
  
  #if defined(DRD_GENERIC_USE_MBED_RP2040)
    #undef DRD_GENERIC_USE_MBED_RP2040
  #endif
  #define DRD_GENERIC_USE_MBED_RP2040      true
   
  #if defined(DRD_GENERIC_USE_EEPROM)
    #undef DRD_GENERIC_USE_EEPROM
  #endif
  #define DRD_GENERIC_USE_EEPROM    false
  
  #if (DRD_GENERIC_DEBUG)
    #warning Use MBED RP2040 (such as NANO_RP2040_CONNECT, RASPBERRY_PI_PICO) and LittleFS
  #endif
  
///////////////////////////// 
#elif ( ( defined(ARDUINO_PORTENTA_H7_M7) || defined(ARDUINO_PORTENTA_H7_M4) ) && defined(ARDUINO_ARCH_MBED) )

  #if defined(BOARD_NAME)
    #undef BOARD_NAME
  #endif

  #if defined(CORE_CM7)
    #if (DRD_GENERIC_DEBUG)
      #warning Using Portenta H7 M7 core
    #endif
    
    #define BOARD_NAME              "PORTENTA_H7_M7"
  #else
    #if (DRD_GENERIC_DEBUG)
      #warning Using Portenta H7 M4 core
    #endif

    #define BOARD_NAME              "PORTENTA_H7_M4"
  #endif
  
  // For Arduino' arduino-mbed core
  // To check and determine if we need to init LittleFS here
  #if MBED_PORTENTA_H7_INITIALIZED
    #define DRD_MBED_LITTLEFS_NEED_INIT     false
    
    #if (DRD_GENERIC_DEBUG)
      #warning MBED_PORTENTA_H7_INITIALIZED in another place
    #endif
    
  #else
    // Better to delay until init done
    #if defined(MBED_PORTENTA_H7_INITIALIZED)
      #undef MBED_PORTENTA_H7_INITIALIZED
    #endif
    #define MBED_PORTENTA_H7_INITIALIZED          true
    
    #define DRD_PORTENTA_LITTLEFS_NEED_INIT       true
  #endif
  
  #if defined(DRD_GENERIC_USE_MBED_PORTENTA)
    #undef DRD_GENERIC_USE_MBED_PORTENTA
  #endif
  #define DRD_GENERIC_USE_MBED_PORTENTA           true
   
  #if defined(DRD_GENERIC_USE_EEPROM)
    #undef DRD_GENERIC_USE_EEPROM
  #endif
  #define DRD_GENERIC_USE_EEPROM    false
  
  #if (DRD_GENERIC_DEBUG)
    #warning Use MBED PORTENTA_H7 and LittleFS
  #endif
  
///////////////////////////// 
#elif ( defined(ARDUINO_ARCH_NRF52840) && defined(ARDUINO_ARCH_MBED) && defined(ARDUINO_ARDUINO_NANO33BLE) )

  // For Arduino' arduino-mbed core
  // To check and determine if we need to init LittleFS here
  #if NANO33BLE_INITIALIZED
    #define DRD_NANO33BLE_NEED_INIT     false
    
    #if (DRD_GENERIC_DEBUG)
      #warning NANO33BLE_INITIALIZED in another place
    #endif
    
  #else
    // Better to delay until init done
    #if defined(NANO33BLE_INITIALIZED)
      #undef NANO33BLE_INITIALIZED
    #endif
    #define NANO33BLE_INITIALIZED           true
    
    #define DRD_NANO33BLE_NEED_INIT     true
  #endif
  
  #if defined(DRD_GENERIC_USE_NANO33BLE)
    #undef DRD_GENERIC_USE_NANO33BLE
  #endif
  #define DRD_GENERIC_USE_NANO33BLE      true
  
  #if defined(DRD_GENERIC_USE_EEPROM)
    #undef DRD_GENERIC_USE_EEPROM
  #endif
  #define DRD_GENERIC_USE_EEPROM    false
  
  #if (DRD_GENERIC_DEBUG)
    #warning Use MBED nRF52840 (such as Nano_33_BLE, Nano_33_BLE_Sense) and LittleFS
  #endif
  
/////////////////////////////
#elif defined(CONFIG_PLATFORM_8721D)

  #if defined(DRD_GENERIC_USE_RTL8720)
    #undef DRD_GENERIC_USE_RTL8720
  #endif
  #define DRD_GENERIC_USE_RTL8720      true
  #if defined(DRD_GENERIC_USE_EEPROM)
    #undef DRD_GENERIC_USE_EEPROM
  #endif
  #define DRD_GENERIC_USE_EEPROM    false
  
  #if (DRD_GENERIC_DEBUG)
    #warning Use RTL8720 and FlashStorage_RTL8720
  #endif
  
#elif ( defined(STM32F0) || defined(STM32F1)  || defined(STM32F2) || defined(STM32F3)  ||defined(STM32F4) || defined(STM32F7) || \
       defined(STM32L0) || defined(STM32L1)  || defined(STM32L4) || defined(STM32H7)  ||defined(STM32G0) || defined(STM32G4) || \
       defined(STM32WB) || defined(STM32MP1) || defined(STM32L5) )  


  #if defined(DRD_GENERIC_USE_STM32)
    #undef DRD_GENERIC_USE_STM32
  #endif
  #define DRD_GENERIC_USE_STM32      true
  #if defined(DRD_GENERIC_USE_EEPROM)
    #undef DRD_GENERIC_USE_EEPROM
  #endif
  #define DRD_GENERIC_USE_EEPROM    false
  
  #if (DRD_GENERIC_DEBUG)
    #warning Use STM32 and FlashStorage_STM32
  #endif
  
/////////////////////////////   
#else
  #if defined(CORE_TEENSY)
    #if (DRD_GENERIC_DEBUG)
      #warning Use TEENSY and EEPROM
    #endif
    
  #elif ( defined(ARDUINO_AVR_ADK) || defined(ARDUINO_AVR_BT) || defined(ARDUINO_AVR_DUEMILANOVE) || defined(ARDUINO_AVR_ESPLORA) \
      || defined(ARDUINO_AVR_ETHERNET) || defined(ARDUINO_AVR_FIO) || defined(ARDUINO_AVR_GEMMA) || defined(ARDUINO_AVR_LEONARDO) \
      || defined(ARDUINO_AVR_LILYPAD) || defined(ARDUINO_AVR_LILYPAD_USB) || defined(ARDUINO_AVR_MEGA) || defined(ARDUINO_AVR_MEGA2560) \
      || defined(ARDUINO_AVR_MICRO) || defined(ARDUINO_AVR_MINI) || defined(ARDUINO_AVR_NANO) || defined(ARDUINO_AVR_NG) \
      || defined(ARDUINO_AVR_PRO) || defined(ARDUINO_AVR_ROBOT_CONTROL) || defined(ARDUINO_AVR_ROBOT_MOTOR) || defined(ARDUINO_AVR_UNO) \
      || defined(ARDUINO_AVR_YUN) )    
      
    #if (DRD_GENERIC_DEBUG)
      #warning Use AVR and EEPROM
    #endif      
    
  #else
  
    #if (DRD_GENERIC_DEBUG)
      #warning Use Unknown board and EEPROM
    #endif
    
  #endif
#endif
///////////////////////////// 
 
//default to use EEPROM, otherwise, use DueFlashStorage or FlashStorage_SAMD
/////////////////////////////
#if DRD_GENERIC_USE_EEPROM
  #include <EEPROM.h>

  #define  FLAG_DATA_SIZE     4

  #ifndef DRD_EEPROM_SIZE
    // Please change according to your application to avoid overwriting or being overwritten
    #define DRD_EEPROM_SIZE     (E2END + 1)
  #endif

/////////////////////////////
#elif DRD_GENERIC_USE_SAMD
  // Include EEPROM-like API for FlashStorage
  #include <FlashStorage_SAMD.h>             //https://github.com/khoih-prog/FlashStorage_SAMD
  
/////////////////////////////  
#elif DRD_GENERIC_USE_SAM_DUE
  //Use DueFlashStorage to simulate EEPROM
  #include <DueFlashStorage.h>                 //https://github.com/sebnil/DueFlashStorage
  DueFlashStorage dueFlashStorage;

/////////////////////////////
#elif DRD_GENERIC_USE_NRF52
  // Include LittleFS similar to SPIFFS
  #include <Adafruit_LittleFS.h>
  #include <InternalFileSystem.h>
  using namespace Adafruit_LittleFS_Namespace;
  
  File DRD_file(InternalFS);

/////////////////////////////
#elif DRD_GENERIC_USE_RP2040

  //Use LittleFS for RPI Pico
  #include <FS.h>
  #include <LittleFS.h>

  FS* filesystem =      &LittleFS;
  #define FileFS        LittleFS

/////////////////////////////
#elif (DRD_GENERIC_USE_MBED_RP2040 && DRD_MBED_LITTLEFS_NEED_INIT)

  //Use LittleFS for MBED RPI Pico
  #include "FlashIAPBlockDevice.h"
  #include "LittleFileSystem.h"
  #include "mbed.h"

  #include <stdio.h>
  #include <errno.h>
  #include <functional>

  #include "BlockDevice.h"

  #if !defined(RP2040_FLASH_SIZE)
    #define RP2040_FLASH_SIZE         (2 * 1024 * 1024)
  #endif

  #if !defined(RP2040_FS_LOCATION_END)
    #define RP2040_FS_LOCATION_END    RP2040_FLASH_SIZE
  #endif

  #if !defined(RP2040_FS_SIZE_KB)
    // Using default 64KB for LittleFS
    #define RP2040_FS_SIZE_KB       (64)
  #endif

  #if !defined(RP2040_FS_START)
    #define RP2040_FS_START           (RP2040_FLASH_SIZE - (RP2040_FS_SIZE_KB * 1024))
  #endif

  #if !defined(FORCE_REFORMAT)
    #define FORCE_REFORMAT            false
  #elif FORCE_REFORMAT
    #warning FORCE_REFORMAT enable. Are you sure ?
  #endif

  FlashIAPBlockDevice bd(XIP_BASE + RP2040_FS_START, (RP2040_FS_SIZE_KB * 1024));

  mbed::LittleFileSystem fs("fs");
  
  #if defined(DRD_FILENAME)
    #undef DRD_FILENAME
  #endif
  #define  DRD_FILENAME     "/fs/drd.dat"
  
  #if (DRD_GENERIC_DEBUG)
    #warning DRD_MBED_LITTLEFS INITIALIZED locally in DoubleResetDetector_Generic
  #endif
    
/////////////////////////////
#elif (DRD_GENERIC_USE_MBED_PORTENTA && DRD_PORTENTA_LITTLEFS_NEED_INIT)

  //Use LittleFS for MBED RPI Pico
  #include "FlashIAPBlockDevice.h"
  #include "LittleFileSystem.h"
  #include "mbed.h"

  #include <stdio.h>
  #include <errno.h>
  #include <functional>

  #include "BlockDevice.h"
  
  #include "mbed_portenta/FlashIAPLimits.h"

  #if !defined(FORCE_REFORMAT)
    #define FORCE_REFORMAT            false
  #elif FORCE_REFORMAT
    #warning FORCE_REFORMAT enable. Are you sure ?
  #endif

  static FlashIAPBlockDevice* blockDevicePtr;

  mbed::LittleFileSystem fs("littlefs");
  
  struct FlashIAPLimits _flashIAPLimits;
  
  #if defined(DRD_FILENAME)
    #undef DRD_FILENAME
  #endif
  #define  DRD_FILENAME     "/littlefs/drd.dat"
  
  #if (DRD_GENERIC_DEBUG)
    #warning DRD_PORTENTA_LITTLEFS INITIALIZED locally in DoubleResetDetector_Generic
  #endif
  
/////////////////////////////
#elif (DRD_GENERIC_USE_NANO33BLE && DRD_NANO33BLE_NEED_INIT)

  //Use LittleFS for MBED Nano33BLE
  #include "FlashIAPBlockDevice.h"
  #include "LittleFileSystem.h"
  #include "mbed.h"

  #include <stdio.h>
  #include <errno.h>
  #include <functional>

  #include "BlockDevice.h"

  #if !defined(NANO33BLE_FLASH_SIZE)
    // Using max 512KB for FS
    #define NANO33BLE_FLASH_SIZE         (1 * 1024 * 1024)
  #endif

  #if !defined(NANO33BLE_FS_LOCATION_END)
    #define NANO33BLE_FS_LOCATION_END    NANO33BLE_FLASH_SIZE
  #endif

  #if !defined(NANO33BLE_FS_SIZE_KB)
    // Using default 64KB for FS
    #define NANO33BLE_FS_SIZE_KB       (64)
  #endif

  #if !defined(NANO33BLE_FS_START)
    #define NANO33BLE_FS_START           (NANO33BLE_FLASH_SIZE - (NANO33BLE_FS_SIZE_KB * 1024))
  #endif

  #if !defined(FORCE_REFORMAT)
    #define FORCE_REFORMAT            false
  #elif FORCE_REFORMAT
    #warning FORCE_REFORMAT enable. Are you sure ?
  #endif

  // nRF52840 flash address from 0, length 1MB.
  // Better to use max half of flash for LitleFS, must be 0x80000 (512KB)
  // FLASH_BASE must be 0x80000, or crash !!!????
  #define FLASH_BASE            0x80000

  FlashIAPBlockDevice wholeBD(FLASH_BASE, 0x80000);
  FlashIAPBlockDevice bd(FLASH_BASE, (NANO33BLE_FS_SIZE_KB * 1024));
  
  mbed::LittleFileSystem fs("littlefs");
  
  #if defined(DRD_FILENAME)
    #undef DRD_FILENAME
  #endif
  #define  DRD_FILENAME     "/littlefs/drd.dat"
  
  #if (DRD_GENERIC_DEBUG)
    #warning DRD_NANO33BLE_LITTLEFS INITIALIZED locally in DoubleResetDetector_Generic
  #endif
    
/////////////////////////////
#elif DRD_GENERIC_USE_STM32

  /////////////////////////////////////////////
      
  #if defined(DATA_EEPROM_BASE)
      // For STM32 devices having integrated EEPROM.
      #include <EEPROM.h>
      
      #if (DRD_GENERIC_DEBUG)
        #warning STM32 devices have integrated EEPROM. Not using buffered API.
      #endif
         
  #else
      /**
       Most STM32 devices don't have an integrated EEPROM. To emulate a EEPROM, the STM32 Arduino core emulated
       the operation of an EEPROM with the help of the embedded flash.
       Writing to a flash is very expensive operation, since a whole flash page needs to be written, even if you only
       want to access the flash byte-wise.
       The STM32 Arduino core provides a buffered access API to the emulated EEPROM. The library has allocated the
       buffer even if you don't use the buffered API, so it's strongly suggested to use the buffered API anyhow.
       */
      #if ( defined(STM32F1xx) || defined(STM32F3xx) )
        #include <FlashStorage_STM32F1.h>       // https://github.com/khoih-prog/FlashStorage_STM32
        
        #if (DRD_GENERIC_DEBUG)
          #warning STM32F1/F3 devices have no integrated EEPROM. Using buffered API with FlashStorage_STM32F1 library
        #endif
        
      #else
        #include <FlashStorage_STM32.h>       // https://github.com/khoih-prog/FlashStorage_STM32
        
        #if (DRD_GENERIC_DEBUG)
          #warning STM32 devices have no integrated EEPROM. Using buffered API with FlashStorage_STM32 library
        #endif
        
      #endif
  #endif    // #if defined(DATA_EEPROM_BASE)

  //////////////////////////////////////////////


#elif DRD_GENERIC_USE_RTL8720
  // Include FlashStorage API for FlashStorage_RTL8720
  #include <FlashStorage_RTL8720.h>             //https://github.com/khoih-prog/FlashStorage_RTL8720

///////////////////////////// 
#endif    //#if DRD_GENERIC_USE_EEPROM

/////////////////////////////////////////////

#define DOUBLERESETDETECTOR_GENERIC_FLAG_SET    0xD0D01234
#define DOUBLERESETDETECTOR_GENERIC_FLAG_CLEAR  0xD0D04321

/////////////////////////////////////////////

class DoubleResetDetector_Generic
{
  public:
  
    /////////////////////////////////////////////
    
    DoubleResetDetector_Generic(unsigned long timeout, int address)
    {
      this->timeout = timeout * 1000;
      this->DRD_EEPROM_START = address;
      doubleResetDetected = false;
      waitingForDoubleReset = false;

/////////////////////////////
#if (DRD_GENERIC_USE_EEPROM)

      EEPROM.begin();
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("\nEEPROM size = ");
      Serial.print(DRD_EEPROM_SIZE);
      Serial.print(", start = ");
      Serial.println(DRD_EEPROM_START);
  #endif

/////////////////////////////
#elif (DRD_GENERIC_USE_STM32)

  #if defined(DATA_EEPROM_BASE)      
      EEPROM.begin();
  #endif   
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("\n(Emulated-)EEPROM size = ");
      Serial.print(EEPROM.length());
      Serial.print(", start = ");
      Serial.println(DRD_EEPROM_START);
  #endif

/////////////////////////////        
#elif DRD_GENERIC_USE_SAMD
      // Do something to init FlashStorage
      
/////////////////////////////      
#elif DRD_GENERIC_USE_SAM_DUE
      // Do something to init DueFlashStorage

/////////////////////////////      
#elif DRD_GENERIC_USE_NRF52
      // Do something to init LittleFS / InternalFS
      // Initialize Internal File System
      InternalFS.begin();

/////////////////////////////      
#elif DRD_GENERIC_USE_RP2040

      bool beginOK = FileFS.begin();

  #if (DRD_GENERIC_DEBUG)      
      if (!beginOK)
      {
        Serial.println("\nLittleFS error");
      }
  #endif
  
/////////////////////////////
#elif (DRD_GENERIC_USE_MBED_RP2040 && DRD_MBED_LITTLEFS_NEED_INIT)

      Serial.print("LittleFS size (KB) = ");
      Serial.println(RP2040_FS_SIZE_KB);

#if FORCE_REFORMAT
      fs.reformat(&bd);
#endif  
      
      int err = fs.mount(&bd);
      
  #if (DRD_GENERIC_DEBUG) 
      Serial.println(err ? "LittleFS Mount Fail" : "LittleFS Mount OK");
  #endif
  
      if (err)
      {
  #if (DRD_GENERIC_DEBUG)     
        // Reformat if we can't mount the filesystem
        Serial.println("Formatting... ");
        Serial.flush();
  #endif
  
        err = fs.reformat(&bd);
      }
  
      bool beginOK = (err == 0);

  #if (DRD_GENERIC_DEBUG)      
      if (!beginOK)
      {
        Serial.println("\nLittleFS error");
      }
  #endif

/////////////////////////////
#elif (DRD_GENERIC_USE_MBED_PORTENTA && DRD_PORTENTA_LITTLEFS_NEED_INIT)

      // Get limits of the the internal flash of the microcontroller
      _flashIAPLimits = getFlashIAPLimits();
      
      Serial.print("Flash Size: (KB) = "); Serial.println(_flashIAPLimits.flash_size / 1024.0);
      Serial.print("FlashIAP Start Address: = 0x"); Serial.println(_flashIAPLimits.start_address, HEX);
      Serial.print("LittleFS size (KB) = "); Serial.println(_flashIAPLimits.available_size / 1024.0);
            
      blockDevicePtr = new FlashIAPBlockDevice(_flashIAPLimits.start_address, _flashIAPLimits.available_size);
      
      if (!blockDevicePtr)
      {
  #if (DRD_GENERIC_DEBUG)       
        Serial.println("Error init FlashIAPBlockDevice");
  #endif      
        return;
      }
      
  #if FORCE_REFORMAT
      fs.reformat(blockDevicePtr);
  #endif

      int err = fs.mount(blockDevicePtr);
      
  #if (DRD_GENERIC_DEBUG) 
      Serial.println(err ? "LittleFS Mount Fail" : "LittleFS Mount OK");
  #endif
  
      if (err)
      {
  #if (DRD_GENERIC_DEBUG)     
        // Reformat if we can't mount the filesystem
        Serial.println("Formatting... ");
  #endif
  
        err = fs.reformat(blockDevicePtr);
      }
  
      bool beginOK = (err == 0);

  #if (DRD_GENERIC_DEBUG)      
      if (!beginOK)
      {
        Serial.println("\nLittleFS error");
      }
  #endif

/////////////////////////////
#elif (DRD_GENERIC_USE_NANO33BLE && DRD_NANO33BLE_NEED_INIT)

      Serial.print("LittleFS size (KB) = ");
      Serial.println(NANO33BLE_FS_SIZE_KB);
  
#if FORCE_REFORMAT
      mbed::LittleFileSystem::format(&bd);
#endif  
      
      int err = fs.mount(&bd);
      
  #if (DRD_GENERIC_DEBUG) 
      Serial.println(err ? "LittleFS Mount Fail" : "LittleFS Mount OK");
  #endif
  
      if (err)
      {
  #if (DRD_GENERIC_DEBUG)     
        // Reformat if we can't mount the filesystem
        Serial.println("Formatting... ");
  #endif
  
        err = mbed::LittleFileSystem::format(&bd);
      }
  
      bool beginOK = (err == 0);

  #if (DRD_GENERIC_DEBUG)      
      if (!beginOK)
      {
        Serial.println("\nLittleFS error");
      }
  #endif
  
/////////////////////////////        
#elif DRD_GENERIC_USE_RTL8720
      // Do something to init FlashStorage_RTL8720
      
/////////////////////////////    
#else
      #error Un-identifiable board selected. Please check your Tools->Board setting.
#endif
    };
    
    /////////////////////////////////////////////

    bool detectDoubleReset()
    {
      doubleResetDetected = detectRecentlyResetFlag();

      if (doubleResetDetected)
      {
#if (DRD_GENERIC_DEBUG)
        Serial.println("doubleResetDetected");
#endif

        clearRecentlyResetFlag();
      }
      else
      {
#if (DRD_GENERIC_DEBUG)
        Serial.println("No doubleResetDetected");
#endif

        setRecentlyResetFlag();
        waitingForDoubleReset = true;
      }

      return doubleResetDetected;

    };
    
    bool waitingForDRD()
    {
      return waitingForDoubleReset;
    }
    
    /////////////////////////////////////////////

    void loop()
    {
      if ( waitingForDoubleReset && (millis() > timeout) )
      {
#if (DRD_GENERIC_DEBUG)
        Serial.println("Stop doubleResetDetecting");
#endif

        stop();
      }
    };

    /////////////////////////////////////////////
    
    void stop()
    {
      clearRecentlyResetFlag();
      waitingForDoubleReset = false;
    };
    
    /////////////////////////////////////////////

    bool doubleResetDetected;
    
    /////////////////////////////////////////////


  private:
    uint32_t DOUBLERESETDETECTOR_FLAG;
    unsigned long timeout;
    int DRD_EEPROM_START;
    bool waitingForDoubleReset;
    
    /////////////////////////////////////////////

#if (DRD_GENERIC_USE_SAMD)

    /////////////////////////////////////////////

    uint32_t readFlagSAMD()
    {        
      uint16_t offset   = DRD_EEPROM_START;               
      uint8_t* _pointer = (uint8_t *) &DOUBLERESETDETECTOR_FLAG;
      
      for (unsigned int i = 0; i < sizeof(DOUBLERESETDETECTOR_FLAG); i++, _pointer++, offset++)
      {              
        *_pointer = EEPROM.read(offset);
      }
      
      return DOUBLERESETDETECTOR_FLAG;
    }
    
    /////////////////////////////////////////////
    
#elif (DRD_GENERIC_USE_SAM_DUE)

    /////////////////////////////////////////////

    uint32_t readFlagSAM_DUE()
    {           
      byte* dataPointer = (byte* ) dueFlashStorage.readAddress(DRD_EEPROM_START);
      
      memcpy(&DOUBLERESETDETECTOR_FLAG, dataPointer, sizeof(DOUBLERESETDETECTOR_FLAG));  
      
      return DOUBLERESETDETECTOR_FLAG;
    }
    
    /////////////////////////////////////////////
    
#elif DRD_GENERIC_USE_NRF52

    /////////////////////////////////////////////

    uint32_t readFlagNRF52()
    {           
      DRD_file.open(DRD_FILENAME, FILE_O_READ);
      
      if (DRD_file)
      {
        DRD_file.seek(DRD_FLAG_OFFSET);
        DRD_file.read((char *) &DOUBLERESETDETECTOR_FLAG, sizeof(DOUBLERESETDETECTOR_FLAG));

  #if (DRD_GENERIC_DEBUG)
        Serial.println("LittleFS Flag read = 0x" + String(DOUBLERESETDETECTOR_FLAG, HEX) );
  #endif

        DRD_file.close(); 
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Loading DRD file failed");
  #endif
      }
           
      return DOUBLERESETDETECTOR_FLAG;
    }
    
    /////////////////////////////////////////////
    
#elif DRD_GENERIC_USE_RP2040

    /////////////////////////////////////////////

    uint32_t readFlagRP2040()
    {           
      File file = FileFS.open(DRD_FILENAME, "r");
      
      if (file)
      {
        file.seek(DRD_FLAG_OFFSET);
        file.read((uint8_t *) &DOUBLERESETDETECTOR_FLAG, sizeof(DOUBLERESETDETECTOR_FLAG));

  #if (DRD_GENERIC_DEBUG)
        Serial.println("LittleFS Flag read = 0x" + String(DOUBLERESETDETECTOR_FLAG, HEX) );
  #endif

        file.close(); 
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Loading DRD file failed");
  #endif
      }
           
      return DOUBLERESETDETECTOR_FLAG;
    }
    
    /////////////////////////////////////////////

#elif DRD_GENERIC_USE_MBED_RP2040

    /////////////////////////////////////////////

    uint32_t readFlagMbedRP2040()
    {           
      FILE *file = fopen(DRD_FILENAME, "r");
      
      if (file)
      {
        fseek(file, DRD_FLAG_OFFSET, SEEK_SET);
        fread((uint8_t *) &DOUBLERESETDETECTOR_FLAG, 1, sizeof(DOUBLERESETDETECTOR_FLAG), file);

  #if (DRD_GENERIC_DEBUG)
        Serial.println("LittleFS Flag read = 0x" + String(DOUBLERESETDETECTOR_FLAG, HEX) );
  #endif

        fclose(file);
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Loading DRD file failed");
  #endif
      }
           
      return DOUBLERESETDETECTOR_FLAG;
    }


    /////////////////////////////////////////////
    
#elif (DRD_GENERIC_USE_MBED_PORTENTA)

    /////////////////////////////////////////////
    
    uint32_t readFlagMbedPortenta()
    {           
      FILE *file = fopen(DRD_FILENAME, "r");
      
      if (file)
      {
        fseek(file, DRD_FLAG_OFFSET, SEEK_SET);
        fread((uint8_t *) &DOUBLERESETDETECTOR_FLAG, 1, sizeof(DOUBLERESETDETECTOR_FLAG), file);

  #if (DRD_GENERIC_DEBUG)
        Serial.println("LittleFS Flag read = 0x" + String(DOUBLERESETDETECTOR_FLAG, HEX) );
  #endif

        fclose(file);
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Loading DRD file failed");
  #endif
      }
           
      return DOUBLERESETDETECTOR_FLAG;
    } 
    
    
    /////////////////////////////////////////////

#elif DRD_GENERIC_USE_NANO33BLE

    /////////////////////////////////////////////

    uint32_t readFlagNano33BLE()
    {           
      FILE *file = fopen(DRD_FILENAME, "r");
      
      if (file)
      {
        fseek(file, DRD_FLAG_OFFSET, SEEK_SET);
        fread((uint8_t *) &DOUBLERESETDETECTOR_FLAG, 1, sizeof(DOUBLERESETDETECTOR_FLAG), file);

  #if (DRD_GENERIC_DEBUG)
        Serial.println("LittleFS Flag read = 0x" + String(DOUBLERESETDETECTOR_FLAG, HEX) );
  #endif

        fclose(file);
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Loading DRD file failed");
  #endif
      }
           
      return DOUBLERESETDETECTOR_FLAG;
    }    
    /////////////////////////////////////////////

#elif (DRD_GENERIC_USE_RTL8720)

    /////////////////////////////////////////////

    uint32_t readFlagRTL8720()
    {             
      // Using name DRD_EEPROM_START, but actually FlashStorage not EEPROM
      FlashStorage.get(DRD_EEPROM_START, DOUBLERESETDETECTOR_FLAG);
      
      return DOUBLERESETDETECTOR_FLAG;
    }
    
    /////////////////////////////////////////////

#endif

    /////////////////////////////////////////////
      
    bool detectRecentlyResetFlag()
    {
/////////////////////////////    
#if (DRD_GENERIC_USE_EEPROM || DRD_GENERIC_USE_STM32)
      EEPROM.get(DRD_EEPROM_START, DOUBLERESETDETECTOR_FLAG);
      doubleResetDetectorFlag = DOUBLERESETDETECTOR_FLAG;

/////////////////////////////
#elif (DRD_GENERIC_USE_SAMD)
      // SAMD code  
      doubleResetDetectorFlag = readFlagSAMD();

/////////////////////////////         
#elif (DRD_GENERIC_USE_SAM_DUE)
      // SAM DUE code    
      doubleResetDetectorFlag = readFlagSAM_DUE();
      
/////////////////////////////      
#elif DRD_GENERIC_USE_NRF52
      // nRF52 code    
      doubleResetDetectorFlag = readFlagNRF52(); 

/////////////////////////////      
#elif DRD_GENERIC_USE_RP2040      
      // RP2040 code    
      doubleResetDetectorFlag = readFlagRP2040(); 

/////////////////////////////
#elif DRD_GENERIC_USE_MBED_RP2040

      // MBED RP2040 code    
      doubleResetDetectorFlag = readFlagMbedRP2040();

/////////////////////////////
#elif (DRD_GENERIC_USE_MBED_PORTENTA)

      // MBED Portenta code    
      doubleResetDetectorFlag = readFlagMbedPortenta();
    
/////////////////////////////
#elif DRD_GENERIC_USE_NANO33BLE

      // MBED Nano_33_BLE code    
      doubleResetDetectorFlag = readFlagNano33BLE();
      
/////////////////////////////
#elif (DRD_GENERIC_USE_RTL8720)
      // RTL8720 code  
      doubleResetDetectorFlag = readFlagRTL8720();
            
#endif    //(DRD_GENERIC_USE_EEPROM || DRD_GENERIC_USE_STM32)
/////////////////////////////

#if (DRD_GENERIC_DEBUG)
      Serial.println("Flag read = 0x" + String(DOUBLERESETDETECTOR_FLAG, HEX) );
#endif

      doubleResetDetected = (doubleResetDetectorFlag == DOUBLERESETDETECTOR_GENERIC_FLAG_SET);
      return doubleResetDetected;
    };
    
    /////////////////////////////////////////////

    void setRecentlyResetFlag()
    {
      doubleResetDetectorFlag = DOUBLERESETDETECTOR_GENERIC_FLAG_SET;

      DOUBLERESETDETECTOR_FLAG = DOUBLERESETDETECTOR_GENERIC_FLAG_SET;

/////////////////////////////
#if (DRD_GENERIC_USE_EEPROM || DRD_GENERIC_USE_STM32)
      EEPROM.put(DRD_EEPROM_START, DOUBLERESETDETECTOR_FLAG);

  #if (DRD_GENERIC_DEBUG)
      delay(1000);
      EEPROM.get(DRD_EEPROM_START, DOUBLERESETDETECTOR_FLAG);
  #endif

/////////////////////////////
#elif (DRD_GENERIC_USE_SAMD)
      // SAMD code     
      uint16_t offset   = DRD_EEPROM_START;               
      uint8_t* _pointer = (uint8_t *) &DOUBLERESETDETECTOR_FLAG;
      
      for (unsigned int i = 0; i < sizeof(DOUBLERESETDETECTOR_FLAG); i++, _pointer++, offset++)
      {              
        EEPROM.write(offset, *_pointer);
      }
      
      EEPROM.commit();
      
  #if (DRD_GENERIC_DEBUG)
      delay(1000);
      readFlagSAMD();
  #endif      

/////////////////////////////
#elif (DRD_GENERIC_USE_SAM_DUE)
      // SAM DUE code           
      dueFlashStorage.write(DRD_EEPROM_START, (byte *) &DOUBLERESETDETECTOR_FLAG, sizeof(DOUBLERESETDETECTOR_FLAG));
      
  #if (DRD_GENERIC_DEBUG)
      delay(1000);
      readFlagSAM_DUE();
  #endif

/////////////////////////////
#elif DRD_GENERIC_USE_NRF52

      // nRF52 code
      DRD_file.open(DRD_FILENAME, FILE_O_WRITE);
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (DRD_file)
      {
        DRD_file.seek(DRD_FLAG_OFFSET);
        DRD_file.write((uint8_t *) &DOUBLERESETDETECTOR_FLAG, sizeof(DOUBLERESETDETECTOR_FLAG));
        DRD_file.close();
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }

/////////////////////////////    
#elif DRD_GENERIC_USE_RP2040

      // RP2040 code
      File file = FileFS.open(DRD_FILENAME, "w");  
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (file)
      {
        file.seek(DRD_FLAG_OFFSET);
        file.write((uint8_t *) &DOUBLERESETDETECTOR_FLAG, sizeof(DOUBLERESETDETECTOR_FLAG));
        file.close();
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }

/////////////////////////////  
#elif DRD_GENERIC_USE_MBED_RP2040

      // Mbed RP2040 code
      FILE *file = fopen(DRD_FILENAME, "w");
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (file)
      {
        fseek(file, DRD_FLAG_OFFSET, SEEK_SET);
        fwrite((uint8_t *) &DOUBLERESETDETECTOR_FLAG, 1, sizeof(DOUBLERESETDETECTOR_FLAG), file);
        
        fclose(file);
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }


/////////////////////////////
#elif (DRD_GENERIC_USE_MBED_PORTENTA)

      // Mbed Portenta code
      FILE *file = fopen(DRD_FILENAME, "w");
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (file)
      {
        fseek(file, DRD_FLAG_OFFSET, SEEK_SET);
        fwrite((uint8_t *) &DOUBLERESETDETECTOR_FLAG, 1, sizeof(DOUBLERESETDETECTOR_FLAG), file);
        
        fclose(file);
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }
      
/////////////////////////////
#elif DRD_GENERIC_USE_NANO33BLE

      // Mbed Nano_33_BLE code
      FILE *file = fopen(DRD_FILENAME, "w");
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (file)
      {
        fseek(file, DRD_FLAG_OFFSET, SEEK_SET);
        fwrite((uint8_t *) &DOUBLERESETDETECTOR_FLAG, 1, sizeof(DOUBLERESETDETECTOR_FLAG), file);
        
        fclose(file);
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }
      
/////////////////////////////
#elif (DRD_GENERIC_USE_RTL8720)
      // RTL8720 code           
      FlashStorage.put(DRD_EEPROM_START, DOUBLERESETDETECTOR_FLAG);
           
  #if (DRD_GENERIC_DEBUG)
      delay(1000);
      readFlagRTL8720();
  #endif  
        
/////////////////////////////               
#endif    //(DRD_GENERIC_USE_EEPROM || DRD_GENERIC_USE_STM32)

#if (DRD_GENERIC_DEBUG)
      Serial.println("SetFlag write = 0x" + String(DOUBLERESETDETECTOR_FLAG, HEX) );
#endif
    };
    
    /////////////////////////////////////////////

    void clearRecentlyResetFlag()
    {
      doubleResetDetectorFlag = DOUBLERESETDETECTOR_GENERIC_FLAG_CLEAR;
      DOUBLERESETDETECTOR_FLAG = DOUBLERESETDETECTOR_GENERIC_FLAG_CLEAR;

/////////////////////////////
      
#if (DRD_GENERIC_USE_EEPROM || DRD_GENERIC_USE_STM32)
      EEPROM.put(DRD_EEPROM_START, DOUBLERESETDETECTOR_FLAG);

  #if (DRD_GENERIC_DEBUG)
      delay(1000);
      EEPROM.get(DRD_EEPROM_START, DOUBLERESETDETECTOR_FLAG);
  #endif
  
/////////////////////////////

#elif (DRD_GENERIC_USE_SAMD)
      // SAMD code     
      uint16_t offset   = DRD_EEPROM_START;               
      uint8_t* _pointer = (uint8_t *) &DOUBLERESETDETECTOR_FLAG;
      
      for (unsigned int i = 0; i < sizeof(DOUBLERESETDETECTOR_FLAG); i++, _pointer++, offset++)
      {              
        EEPROM.write(offset, *_pointer);
      }
      
      EEPROM.commit();
      
  #if (DRD_GENERIC_DEBUG)
      delay(1000);
      readFlagSAMD();
  #endif
  
/////////////////////////////

#elif (DRD_GENERIC_USE_SAM_DUE)
      // SAM DUE code           
      dueFlashStorage.write(DRD_EEPROM_START, (byte *) &DOUBLERESETDETECTOR_FLAG, sizeof(DOUBLERESETDETECTOR_FLAG));
      
  #if (DRD_GENERIC_DEBUG)
      delay(1000);
      readFlagSAM_DUE();
  #endif

/////////////////////////////

#elif DRD_GENERIC_USE_NRF52

      // nRF52 code
      DRD_file.open(DRD_FILENAME, FILE_O_WRITE);
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (DRD_file)
      {
        DRD_file.seek(DRD_FLAG_OFFSET);
        DRD_file.write((uint8_t *) &DOUBLERESETDETECTOR_FLAG, sizeof(DOUBLERESETDETECTOR_FLAG)); 
   
        DRD_file.close();
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }   
      
      delay(1000);
      readFlagNRF52();

/////////////////////////////

#elif DRD_GENERIC_USE_RP2040

      // RP2040 code
      File file = FileFS.open(DRD_FILENAME, "w");
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (file)
      {
        file.seek(DRD_FLAG_OFFSET);
        file.write((uint8_t *) &DOUBLERESETDETECTOR_FLAG, sizeof(DOUBLERESETDETECTOR_FLAG)); 
   
        file.close();
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }   
      
      delay(1000);
      readFlagRP2040();

/////////////////////////////

#elif DRD_GENERIC_USE_MBED_RP2040

      // Mbed RP2040 code
      FILE *file = fopen(DRD_FILENAME, "w");
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (file)
      {       
        fseek(file, DRD_FLAG_OFFSET, SEEK_SET);
        fwrite((uint8_t *) &DOUBLERESETDETECTOR_FLAG, 1, sizeof(DOUBLERESETDETECTOR_FLAG), file);
        
        fclose(file);
        
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }   
      
      delay(1000);
      readFlagMbedRP2040();


/////////////////////////////
#elif (DRD_GENERIC_USE_MBED_PORTENTA)

      // Mbed Portenta code
      FILE *file = fopen(DRD_FILENAME, "w");
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (file)
      {       
        fseek(file, DRD_FLAG_OFFSET, SEEK_SET);
        fwrite((uint8_t *) &DOUBLERESETDETECTOR_FLAG, 1, sizeof(DOUBLERESETDETECTOR_FLAG), file);
        
        fclose(file);
        
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }   
      
      delay(1000);
      readFlagMbedPortenta();

/////////////////////////////
#elif DRD_GENERIC_USE_NANO33BLE

      // Mbed Nano_33_BLE code
      FILE *file = fopen(DRD_FILENAME, "w");
      
  #if (DRD_GENERIC_DEBUG)
      Serial.print("Saving to DRD file : 0x");
      Serial.println(String(DOUBLERESETDETECTOR_FLAG, HEX));
  #endif

      if (file)
      {       
        fseek(file, DRD_FLAG_OFFSET, SEEK_SET);
        fwrite((uint8_t *) &DOUBLERESETDETECTOR_FLAG, 1, sizeof(DOUBLERESETDETECTOR_FLAG), file);
        
        fclose(file);
        
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file OK");
  #endif
      }
      else
      {
  #if (DRD_GENERIC_DEBUG)
        Serial.println("Saving DRD file failed");
  #endif
      }   
      
      delay(1000);
      readFlagNano33BLE();
      
/////////////////////////////

#elif (DRD_GENERIC_USE_RTL8720)
     
      // RTL8720 code           
      FlashStorage.put(DRD_EEPROM_START, DOUBLERESETDETECTOR_FLAG);
           
  #if (DRD_GENERIC_DEBUG)
      delay(1000);
      readFlagRTL8720();
  #endif  
             
#endif    //(DRD_GENERIC_USE_EEPROM || DRD_GENERIC_USE_STM32)

/////////////////////////////

#if (DRD_GENERIC_DEBUG)
      Serial.println("ClearFlag write = 0x" + String(DOUBLERESETDETECTOR_FLAG, HEX) );
#endif
    };
    
    /////////////////////////////////////////////

    uint32_t doubleResetDetectorFlag;
};
#endif // DoubleResetDetector_Generic_H
