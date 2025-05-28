#include "Utils.h"

void initI2C() {
  Wire.begin();  //Start the I2C as master.
  TWBR = 12;
  // The I2C clk freq = F_cpu / (16 + 2* TWBR * (4^TWPS))
  // By default, TWPS = 0
  // F_cpu = 16 MHz for an arduino nano
  // -> I2C's freq is 400 kHz, 4 times higher than default
}

void enterErrorMode(uint8_t R, uint8_t G, uint8_t B) {
  Serial.println("Entering error mode with color : " + String(R) + "," + String(G) + "," + String(B));
  // 1 : Make sure to shut down the motors
  applyTargetSpeed(0, 0, 0, 0);
  // 2 : display the error code. Alternate between white and the error's color
  for (;;) {
    setLEDColor(30, 30, 30);
    delay(500);
    setLEDColor(R, G, B);
    delay(500);
  }
}


void waitWire(int numBytes, float maxTime, uint8_t R, uint8_t G, uint8_t B) {
  float maxDelay = millis() + maxTime;
  while (Wire.available() < numBytes && millis() <= maxDelay) {
  }
  if (Wire.available() < numBytes) {
    enterErrorMode(R, G, B);
  }
}