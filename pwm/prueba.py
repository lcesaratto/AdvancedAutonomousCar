#!/usr/bin/env python

import wiringpi as wpi 
import time

wpi.wiringPiSetup()
wpi.pinMode(2, 1)

while True:
	wpi.digitalWrite(2, 1)
	time.sleep(5)
	wpi.digitalWrite(2, 0)
	time.sleep(5)

