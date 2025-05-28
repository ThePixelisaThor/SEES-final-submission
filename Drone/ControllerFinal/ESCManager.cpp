#include "ESCManager.h"


DShot escM1(DShot::Mode::DSHOT300);
DShot escM2(DShot::Mode::DSHOT300);
DShot escM3(DShot::Mode::DSHOT300);
DShot escM4(DShot::Mode::DSHOT300);


void activateMotorsSequentially() {
  // Turns all the motors on one after the other, for trobleshooting.
  float speed = 800;
  float timeOn = 900;
  unsigned long startTime = millis();
  while (millis() - startTime < timeOn) {
    applyTargetSpeed(0, 0, 0, 0);  // Turn on motor 1
    delay(100);
  }
  delay(500);
  Serial.println("Activating motor sequentially");
  startTime = millis();
  while (millis() - startTime < timeOn) {
    applyTargetSpeed(speed, 0, 0, 0);  // Turn on motor 1
    delay(100);
  }

  startTime = millis();
  while (millis() - startTime < timeOn) {
    applyTargetSpeed(0, speed, 0, 0);  // Turn on motor 2
    delay(100);
  }

  startTime = millis();
  while (millis() - startTime < timeOn) {
    applyTargetSpeed(0, 0, speed, 0);  // Turn on motor 3
    delay(100);
  }

  startTime = millis();
  while (millis() - startTime < timeOn) {
    applyTargetSpeed(0, 0, 0, speed);  // Turn on motor 4
    delay(100);
  }
  Serial.println("Done sequence");
}


void applyTargetSpeed(float speedRotor1, float speedRotor2, float speedRotor3, float speedRotor4) {
  uint16_t target_speedRotor1 = 0;
  uint16_t target_speedRotor2 = 0;
  uint16_t target_speedRotor3 = 0;
  uint16_t target_speedRotor4 = 0;

  float divFactor = 2.384; // To manage the speed of the rotors, as it might change with modifications on the drone or with the battery voltage;
  
  target_speedRotor1 = (uint16_t)min(speedRotor1 / divFactor, 1550); // Caps the maximal target speed
  target_speedRotor2 = (uint16_t)min(speedRotor2 / divFactor, 1550);
  target_speedRotor3 = (uint16_t)min(speedRotor3 / divFactor, 1550);
  target_speedRotor4 = (uint16_t)min(speedRotor4 / divFactor, 1550);

  escM1.setThrottle(target_speedRotor1);
  escM2.setThrottle(target_speedRotor2);
  escM3.setThrottle(target_speedRotor3);
  escM4.setThrottle(target_speedRotor4);
  /*
  escM1.setThrottle(target_speedRotor2);
  escM2.setThrottle(target_speedRotor4);
  escM3.setThrottle(target_speedRotor1);
  escM4.setThrottle(target_speedRotor3);*/
}

void setDirs() {
  // Can be changed in the wiring of one of the motors changes.
  escM1.setDirection(0);
  escM2.setDirection(0);
  escM3.setDirection(0);
  escM4.setDirection(0);
}

void initESC() {
  delay(10);
  // Because the motors are not connected in the same order as what the PID expects, so the mapping is done at this step.
  escM1.attach(6);
  escM2.attach(4);
  escM3.attach(7);
  escM4.attach(5);

  delay(500); // Give some delay to the ESC to be sure the command is executed.

  escM1.setThrottle(0);
  escM2.setThrottle(0);
  escM3.setThrottle(0);
  escM4.setThrottle(0);
}