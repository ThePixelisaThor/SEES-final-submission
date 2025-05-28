#ifndef ESC_MANAGER_H
#define ESC_MANAGER_H

#include <DShot.h>
#include "Arduino.h"
#include "Radio.h"

void applyTargetSpeed(float speedRotor1, float speedRotor2, float speedRotor3, float speedRotor4);
void activateMotorsSequentially();
void initESC();

#endif