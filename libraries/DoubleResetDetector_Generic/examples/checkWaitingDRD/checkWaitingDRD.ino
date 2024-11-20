/****************************************************************************************************************************
   checkWaitingDRD.ino
   DoubleResetDetector_Generic.h
  For AVR Mega, Teensy, STM32, nRF52, SAM DUE, SAMD21, SAMD51 boards

  DoubleResetDetector_Generic is a library for the Arduino AVR, Teensy, SAM-DUE, SAMD, STM32. etc. boards
  to enable trigger configure mode by resetting the boards twice within configurable timeout seconds.

  Based on and modified from DataCute https://github.com/datacute/DoubleResetDetector and
  https://github.com/khoih-prog/ESP_DoubleResetDetector

  Built by Khoi Hoang https://github.com/khoih-prog/DoubleResetDetector_Generic
  Licensed under MIT license
 *****************************************************************************************************************************/

/****************************************************************************************************************************
   This example will open a configuration portal when the reset button is pressed twice.
   This method works well on Wemos boards which have a single reset button on board. It avoids using a pin for launching the configuration portal.

   How It Works
   1) AVR Mega, Teensy
   Save data in EPPROM from address 1020, size 1024 bytes (both configurable)
   Note: Teensy 4.0 has only 1080 bytes of EEPROM-simulated Flash
   2) SAMD
   Save data in EEPROM-simulated FlashStorage from address 0 (configurable to avoid conflict)
   3) SAM DUE
   Save data in DueFlashStorage from address 1020 (configurable to avoid conflict)
   4) Adafruit nRF52-based boards
   Save data in InternalFS, fle "/drd.dat" location 0
   5) RTL8720
   Save data in FlashStorage from address 0 (configurable to avoid conflict)
   6) STM32
   Save data in EEPROM-simulated FlashStorage from address 0 (configurable to avoid conflict)

   So when the device starts up it checks the InternalFS file "/drd.dat", EEPROM or (Due)FlashStorage for a flag to see if it has been
   recently reset within the configurable timeout seconds
   It'll then set a flag, and display a message to signal if the DR is detected

   Settings
   There are two values to be set in the sketch.

   DRD_TIMEOUT - Number of seconds to wait for the second reset. Set to 10s in the example.
   DRD_ADDRESS - The address in ESP8266 RTC RAM to store the flag. This memory must not be used for other purposes in the same sketch. Set to 0 in the example.

   This example, originally relied on the Double Reset Detector library from https://github.com/datacute/DoubleResetDetector
   To support ESP32, use ESP_DoubleResetDetector library from https://github.com/khoih-prog/ESP_DoubleResetDetector
   To support AVR, Teensy, SAM DUE, SAMD and STM32, etc., use this DoubleResetDetector_Generic from //https://github.com/khoih-prog/DoubleResetDetector_Generic
 *****************************************************************************************************************************/


// These defines must be put before #include <DoubleResetDetector_Generic.h>
// to select where to store DoubleResetDetector's variable.
// For ESP32, You must select one to be true (EEPROM or SPIFFS)
// For ESP8266, You must select one to be true (RTC, EEPROM, LITTLEFS or SPIFFS)
// Otherwise, library will use default EEPROM storage

// This example demonstrates how to use new function waitingForDRD() to signal the stage of DRD
// waitingForDRD() returns true if in DRD_TIMEOUT, false when out of DRD_TIMEOUT
// In this example, LED_BUILTIN will blink in DRD_TIMEOUT period, ON when DR has been detected, OFF otherwise

#define DRD_GENERIC_DEBUG       false

// You have to select true for the first time for any board
#define FORCE_REFORMAT          false

// Uncomment to have debug
//#define DOUBLERESETDETECTOR_DEBUG       true

#include <DoubleResetDetector_Generic.h>      //https://github.com/khoih-prog/DoubleResetDetector_Generic

// Number of seconds after reset during which a 
// subsequent reset will be considered a double reset.
#define DRD_TIMEOUT 10

// RTC Memory Address for the DoubleResetDetector to use
#define DRD_ADDRESS 0

DoubleResetDetector_Generic* drd;

// Change according to your board
#ifndef LED_BUILTIN
  // Generic boards following LED at pin 13
  #define LED_BUILTIN       13
#endif

// Change according to your board
#define LED_OFF             LOW
#define LED_ON              HIGH

bool DRD_Detected = false;

void check_status()
{
  static unsigned long checkstatus_timeout  = 0;
  static bool LEDState = LED_OFF;

  static unsigned long current_millis;

#define DRD_CHECK_INTERVAL    500L

  current_millis = millis();

  // If DRD_Detected, don't need to blink, just keep LED_BUILTIN ON
  if ( !DRD_Detected && ((current_millis > checkstatus_timeout) || (checkstatus_timeout == 0)) )
  {
    // If in DRD checking loop, blinking the LED_BUILTIN
    if ( drd->waitingForDRD() )
    {
      digitalWrite(LED_BUILTIN, LEDState);

      LEDState = !LEDState;    
    }
    else
    {
      digitalWrite(LED_BUILTIN, LED_OFF);
    }
    
    checkstatus_timeout = current_millis + DRD_CHECK_INTERVAL;
  }
}

void setup()
{
  pinMode(LED_BUILTIN, OUTPUT);
  
  Serial.begin(115200);
  while (!Serial);

  delay(200);

#if defined(BOARD_NAME)
  Serial.print(F("\nStarting checkWaitingDRD on")); Serial.println(BOARD_NAME);
#else
  Serial.print(F("\nStarting checkWaitingDRD"));
#endif  

  Serial.println(DOUBLERESETDETECTOR_GENERIC_VERSION);
  Serial.println("-----------------------------------");
   
  drd = new DoubleResetDetector_Generic(DRD_TIMEOUT, DRD_ADDRESS);

  if (drd->detectDoubleReset()) 
  {
    Serial.println("Double Reset Detected");
    digitalWrite(LED_BUILTIN, LED_ON);
    DRD_Detected = true;
  } 
  else 
  {
    Serial.println("No Double Reset Detected");
    digitalWrite(LED_BUILTIN, LED_OFF);
  }
}

void loop()
{
  // Call the double reset detector loop method every so often,
  // so that it can recognise when the timeout expires.
  // You can also call drd.stop() when you wish to no longer
  // consider the next reset as a double reset.
  drd->loop();

  check_status();
}
