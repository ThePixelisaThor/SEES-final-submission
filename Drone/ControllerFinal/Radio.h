#ifndef RADIO_H
#define RADIO_H


#include <SPI.h>
#include "Arduino.h"
#include <LoRa.h>

void initLora();
void waitForPckt(unsigned long timeStart);
bool hasCorrectChecksum(char *res, int resLength);
void readRadio(int maxDelay, char *result, int *resLength, int delay2);
unsigned short calculateChecksum(char *data, int length);
char *extractPayload(char *pckt, int delimPos);
String makePacket(String data);
void sendDataByRadio(String toSend);
char *unpackPacket(char *pckt, int length);
void PrintRadioData(char *result, int *resLength) ;

#endif