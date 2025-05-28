#include "RXManager.h"

unsigned long lastPcktReceived = 0;
float throttleLoraInput = 0, sideLoraInput = 0, forward_LORA_input = 0;
float LORA_X2 = 0, loraY2 = 0;


void loraUpdate(FlightStatus *status) {
  char *res;
  res = (char *)malloc(20 * sizeof(char));
  int resLength;

  readRadio(0, res, &resLength, 0);

  if (resLength > 0) {
    if (res[0] == 'F') {
      lastPcktReceived = millis();
      updateTargets(res, resLength, status);
    } else {
      turnOff();  // Corrupted packet received
    }
  }

  if (lastPcktReceived < millis() - 350) {
    resetInputs();
  }
  free(res);
}

void resetInputs() {
  throttleLoraInput = 0;
  sideLoraInput = 0;
  forward_LORA_input = 0;
  LORA_X2 = 0;
  loraY2 = 0;
}

void turnOff() {
  throttleLoraInput = 0;
  sideLoraInput = 0;
  forward_LORA_input = 0;
}

float rawValToAxis(uint16_t rawVal) {
  // Convnerts the raw output of the potentiometer of a joystick into the user's input.
  // Deadzone
  const float CENTER_LOW = 400.0;
  const float CENTER_HIGH = 460.0;
  float rawValF = (float)rawVal;
  if (rawValF > CENTER_LOW && rawValF < CENTER_HIGH) {
    return 0.0;
  }

  if (rawValF <= CENTER_LOW) {
    return rawValF - CENTER_LOW;
  }

  return rawValF - CENTER_HIGH;
}

// The remote's ADC uses 10 bits. However, as this value is then used in a String, it would creaste issues if left at 0.
// As a workaround, and since the 10 bits take 2 bytes anyway, the 10 bits are wrapped with a one on the left and on the right,
// ensuring none of the byte is null. This does the unwrapping.
uint16_t bitsToint(char highByte, char lowByte) {
  uint8_t high = ((uint8_t)highByte & 0x7F);
  uint8_t low = ((uint8_t)lowByte) >> 1;
  return (uint16_t)(low + 128 * high);
}

void updateTargets(char *pckt, int pcktLength, FlightStatus *status) {
  if (pckt == NULL || status == NULL) return;
  if (pcktLength < 9 || pckt[0] != 'F') return;

  char highByte = pckt[1];
  char lowByte = pckt[2];
  sideLoraInput = -1.0 * rawValToAxis(bitsToint(highByte, lowByte));

  highByte = pckt[3];
  lowByte = pckt[4];
  throttleLoraInput = -1.0 * rawValToAxis(bitsToint(highByte, lowByte));

  highByte = (char)pckt[5];
  lowByte = (char)pckt[6];
  LORA_X2 = rawValToAxis(bitsToint(highByte, lowByte));

  highByte = (char)pckt[7];
  lowByte = (char)pckt[8];
  loraY2 = rawValToAxis(bitsToint(highByte, lowByte));

  updateMotorArmingState(status);
}

void updateMotorArmingState(FlightStatus *status) {
  // Move the joystick to the left to go from OFF to READY
  if (*status == FlightStatus::OFF && throttleLoraInput < 10 && sideLoraInput < -25) {
    *status = FlightStatus::READY;
    return;
  }
  // Move it back to the center to go from Ready to ON
  if (*status == FlightStatus::READY && throttleLoraInput < 10 && abs(sideLoraInput) < 10) {
    *status = FlightStatus::ON;
    reinitIMU();
    reinitPID();
    return;
  }

  if (*status == FlightStatus::ON && throttleLoraInput < 5) {
    if (sideLoraInput > 25 || sideLoraInput < -25) {
      *status = FlightStatus::OFF;
    }
  }
}


float getLORAThrottle() {
  return throttleLoraInput;
}

float getLORASide() {
  return sideLoraInput;
}

float getLORAForward() {
  return forward_LORA_input;
}

float getLORAX2() {
  return LORA_X2;
}

float getLORAY2() {
  return loraY2;
}
