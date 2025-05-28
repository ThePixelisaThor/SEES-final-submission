#include <stdint.h>
#ifndef UTILS_H
#define UTILS_H

#define UPDATE_FREQ 255   //  Hz
constexpr int UPDATE_PER = 1000000 / UPDATE_FREQ; // In micro second
#define DEG_TO_RAD (PI / 180.0)
#define RAD_TO_DEG (180.0 / PI)


enum class FlightStatus {
  OFF = 0,
  READY = 1,
  ON = 2
};

#include "Arduino.h"
#include "ESCManager.h"
#include "RGBLedManager.h"
#include <Wire.h>

void initI2C();
void enterErrorMode(uint8_t R, uint8_t G, uint8_t B);
void waitWire(int numBytes, float maxTime, uint8_t R, uint8_t G, uint8_t B);

#endif