#include "ESCManager.h"
#include "RXManager.h"
#include "Utils.h"
#include "RGBLedManager.h"
#include "IMUManager.h"
#include "RXManager.h"
#include "PIDController.h"


FlightStatus status;  // The current status. Off means shut down all the motors. Ready means that it is one joystick movement from take off. On means that the motors now follow the remote's throttle.
unsigned long loopTimer, lastPeriodicUpdateMillis;

void setup() {
  initNeopixel();
  Serial.begin(115200);
  Serial.println("Booting...");
  initLora();
  initI2C();
  initIMU();
  initESC();
  status = FlightStatus::OFF;
  loopTimer = micros();
  Serial.println("Done...");
  lastPeriodicUpdateMillis = millis();  // Comes after the last print to ensure the timing isn't disturbed
}

void loop() {
  readIMU();
  manageTimers(); // For the periodic updates
  updateRotation(); // Update the current estimation of the rotation
  calculatePid();
  updateESCTarget(status);
  endOfLoopDelay();

  if (status == FlightStatus::ON)  // Ensures that the motors turn if and only if the drone is on.
    applyTargetSpeed(getESC1TargetSpeed(), getESC2TargetSpeed(), getESC3TargetSpeed(), getESC4TargetSpeed());
  else
    applyTargetSpeed(0, 0, 0, 0);
}

void endOfLoopDelay() {
  if (micros() - loopTimer >= UPDATE_PER) {
    Serial.println("Error : the loop takes too much time, making the angle update faulty!");
  }

  while (micros() - loopTimer < UPDATE_PER)
    ;

  loopTimer = micros();
}

void manageTimers() {
  if (lastPeriodicUpdateMillis < millis()) {
    periodicUpdate();
    lastPeriodicUpdateMillis = millis() + 50;
  }
}

void periodicUpdate() {
  // Takes care of the things that don't need to be made at every loop, but still need to be done periodically.
  loraUpdate(&status);
  updateLED(status);
}
