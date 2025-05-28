#include "Radio.h"
#include "Utils.h"
#include "RemoteControlLogic.h"

#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include <SoftwareSerial.h>

#define OLED_RESET -1        // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C  ///< 0x3C for 128x32 OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);


volatile byte receivedByte = 0;         // Variable to store received data
volatile boolean dataReceived = false;  // Flag to indicate new data
uint8_t movement_received;

int movementX = 0;
int movementY = 0;

long time_received_last_movement;
#define RX_PIN 6  // Receive pin
#define TX_PIN 7  // (Optional) Transmit pin

SoftwareSerial softSerial(RX_PIN, TX_PIN);  // RX, TX


void setup() {
  initSerial();
  Serial.println("Init...");
  initControlLogic();
  initLora();
  softSerial.begin(115200);
  // When using the screen for debug.
  // if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
  //   Serial.println(F("SSD1306 allocation failed"));
  //   for (;;)
  //     ;
  // }

  // display.clearDisplay();
  // display.setTextColor(WHITE);
  // display.display();
}


void loop() {
  String ctrl = GetControlString(movementY, movementX);
  delay(50);
  sendDataByRadio(ctrl);

  // displayVals();

  if (time_received_last_movement + 2e3 < millis()) {
    movementX = 0;
    movementY = 0;
  }

  if (softSerial.available()) {
    uint8_t newMvmt = (uint8_t)softSerial.read();

    bool valid = (newMvmt == movement_received);
    Serial.println("Received mvmt: ");
    Serial.println(newMvmt);
    dataReceived = false;
    movement_received = newMvmt;
    if (valid) {
      handleMovement();
    }
  }

  Serial.println("Mvm x : " + String(movementX) + " y" + String(movementY));
}

void handleMovement() {
  time_received_last_movement = millis();
  int movValX = 350;
  int movValY = 350;
  applyMovement(movement_received, movValX, movValY);
}

void applyMovement(int movement, int movValX, int movValY) {
  switch (movement) {
    case 0:
    case 1:
    case 2:
    case 7:
      applyNoMovement();
      break;
    case 3:
      applyLeftMovement(movValX);
      break;
    case 4:
      applyRightMovement(movValX);
      break;
    case 5:
      applyForwardMovement(movValY);
      break;
    case 6:
      applyBackwardMovement(movValY);
      break;
    default:
      applyNoMovement();
      break;
  }
}

void applyNoMovement() {
  movementX = 0;
  movementY = 0;
}

void applyLeftMovement(int movValX) {
  movementX = movValX;
  movementY = 0;
}

void applyRightMovement(int movValX) {
  movementX = -movValX;
  movementY = 0;
}

void applyForwardMovement(int movValY) {
  movementX = 0;
  movementY = movValY;
}

void applyBackwardMovement(int movValY) {
  movementX = 0;
  movementY = -movValY;
}


void displayVals() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.cp437(true);

  display.println("X1: " + String(rawValToAxis(xAxis())));
  display.println("Y1: " + String(rawValToAxis(yAxis())));
  display.println("X2: " + String(rawValToAxis(xAxis2())));
  display.println("Y2: " + String(rawValToAxis(yAxis2())));

  display.display();
}
