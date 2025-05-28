#ifndef RADIO_H
#define RADIO_H


#include <SPI.h>
#include <LoRa.h>
#include "Utils.h"

void initLora();
void WaitForPckt(unsigned long timeStart);
bool hasCorrectChecksum(char *res, int resLength);
void ReadRadio(int maxDelay, char *result, int *resLength, int delay2);
unsigned short calculateChecksum(char *data, int length);
char *extractPayload(char *pckt, int delimPos);
String makePacket(String data);
void sendDataByRadio(String toSend);
char *unpackPacket(char *pckt, int length);

#endif