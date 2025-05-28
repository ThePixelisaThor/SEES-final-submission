#ifndef RX_H
#define RX_H

#include "Arduino.h"
#include "Radio.h"
#include "PIDController.h"

void loraUpdate(FlightStatus *status);
void updateTargets(char *pckt, int pcktLength, FlightStatus *status);

float getLORAThrottle();
float getLORASide();
float getLORAForward();

uint16_t bitsToInt(unsigned char high_byte, unsigned char low_byte);

float rawValToAxis(uint16_t rawVal);
void resetInputs();

float getLORAX2();

float getLORAY2();
void updateMotorArmingState(FlightStatus *status);
#endif