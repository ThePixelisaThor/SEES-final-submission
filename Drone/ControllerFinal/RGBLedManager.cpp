#include "RGBLedManager.h"

#define RGBLED_PIN 8
#define NUM_RGB_LEDS 1

Adafruit_NeoPixel pixels(NUM_RGB_LEDS, RGBLED_PIN, NEO_GRB + NEO_KHZ800);

void initNeopixel() {
  pixels.begin();
  // Output a dim white to show that the arduino has booted
  pixels.setPixelColor(0, pixels.Color(10, 10, 10));
  setLEDColor(10, 10, 10);
}

void updateLED(FlightStatus status) {
  if (status == FlightStatus::OFF) {
    setLEDColor(0, 0, 20);
  }
  if (status == FlightStatus::READY) {
    setLEDColor(0, 20, 20);
  }
  if (status == FlightStatus::ON) {
    float throttle_LORA_input = getLORAThrottle();
    uint8_t targetPwr = (uint8_t)abs(throttle_LORA_input / 2.0);
    if (targetPwr > 200) {
      targetPwr = 200;
    }
    targetPwr += 10;
    setLEDColor(0,targetPwr,0);
  }
}

void setLEDColor(uint8_t R, uint8_t G, uint8_t B) {
  pixels.setPixelColor(0, pixels.Color(R, G, B));
  pixels.show();
}