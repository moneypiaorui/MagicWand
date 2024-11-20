# FlashStorage_STM32F1 Library

[![arduino-library-badge](https://www.ardu-badge.com/badge/FlashStorage_STM32F1.svg?)](https://www.ardu-badge.com/FlashStorage_STM32F1)
[![GitHub release](https://img.shields.io/github/release/khoih-prog/FlashStorage_STM32F1.svg)](https://github.com/khoih-prog/FlashStorage_STM32F1/releases)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/khoih-prog/FlashStorage_STM32F1/blob/main/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)
[![GitHub issues](https://img.shields.io/github/issues/khoih-prog/FlashStorage_STM32F1.svg)](http://github.com/khoih-prog/FlashStorage_STM32F1/issues)

---
---

## Table of Contents

* [Changelog](#changelog)
  * [Releases v1.1.0](#releases-v110)
  * [Releases v1.0.1](#releases-v101)
  * [Initial Releases v1.0.0](#initial-releases-v100)

---
---

## Changelog

### Releases v1.1.0

1. Fix `multiple-definitions` linker error.
2. Add example [multiFileProject](examples/multiFileProject) to demo for multiple-file project.
3. Update `Packages' Patches`

### Releases v1.0.1

1. Fix mistake in initial releases


### Initial Releases v1.0.0

1. Initial release to support STM32F1/F3 boards with / without integrated EEPROM
2. Tested using genuine STM32F103, or CH32F103/CS32F103 boards with 32KB, 64KB or 128KB Flash Size
3. Configurable Flash Sector to use
4. FlashStorage size is 1024 bytes (including `1-byte valid` and `4-byte signature`)

