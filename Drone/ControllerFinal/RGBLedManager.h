#ifndef RGB_H
#define RGB_H

#include <Adafruit_NeoPixel.h>
#include "Arduino.h"
#include "RXManager.h"


void initNeopixel();
void updateLED(FlightStatus status);
void setLEDColor(uint8_t R, uint8_t G, uint8_t B);
void turnOff();

#endif