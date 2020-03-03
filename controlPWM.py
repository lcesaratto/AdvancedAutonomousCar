# Simple demo of of the PCA9685 PWM servo/LED controller library.
# This will move channel 0 from min to max position repeatedly.
# Author: Tony DiCola
# License: Public Domain
from __future__ import division
import time

# Import the PCA9685 module.
import Adafruit_PCA9685
import wiringpi as wpi
import keyboard

#wpi.wiringPiSetup()
#wpi.pinMode(3,1)
#wpi.pinMode(4,1)

#wpi.digitalWrite(3,1)
#wpi.digitalWrite(4,1)
# Uncomment to enable debug output.
#import logging
#logging.basicConfig(level=logging.DEBUG)

# Initialise the PCA9685 using the default address (0x40).
servo_min = 0  #0  Min pulse length out of 4096
#servo_max = 4095  # Max pulse length out of 4096

def iniciarPWM():
	pwm = Adafruit_PCA9685.PCA9685()

# Alternatively specify a different address and/or bus:
#pwm = Adafruit_PCA9685.PCA9685(address=0x41, busnum=2)

# Configure min and max servo pulse lengths

# Helper function to make setting a servo pulse width simpler.
	'''def set_servo_pulse(channel, pulse):
	    pulse_length = 1000000    # 1,000,000 us per second
	    pulse_length //= 600       # 60 Hz
	    print('{0}us per period'.format(pulse_length))
	    pulse_length //= 4096     # 12 bits of resolution
	    print('{0}us per bit'.format(pulse_length))
	    pulse *= 1000
	    pulse //= pulse_length
	    pwm.set_pwm(channel, 0, pulse)'''

	# Set frequency to 60hz, good for servos.
	pwm.set_pwm_freq(600)
	return pwm

#Termina funcion init

def forward(pwm, servo_fw):
	pwm.set_pwm(2, 0, servo_min) #Atras Derecha
	pwm.set_pwm(6, 0, servo_min) #Atras Izquierda
	pwm.set_pwm(1, 0, servo_fw) #Delante Derecha
	pwm.set_pwm(5, 0, servo_fw) #Delante Izquierda
	pwm.set_pwm(0, 0, servo_max)
	pwm.set_pwm(4, 0, servo_max)


def backward(pwm, servo_bw):
	pwm.set_pwm(2, 0, servo_bw) #Atras Derecha
	pwm.set_pwm(6, 0, servo_bw) #Atras Izquierda
	pwm.set_pwm(1, 0, servo_min) #Delante Derecha
	pwm.set_pwm(5, 0, servo_min) #Delante Izquierda
	pwm.set_pwm(0, 0, servo_max)
	pwm.set_pwm(4, 0, servo_max)


def giroDerecha(pwm):
	pwm.set_pwm(2, 0, servo_max)
	pwm.set_pwm(6, 0, servo_min)
	pwm.set_pwm(1, 0, servo_min) #Delante.
	pwm.set_pwm(5, 0, servo_max) #Delante.
	

def giroIzquierda(pwm):
	pwm.set_pwm(2, 0, servo_min)
	pwm.set_pwm(6, 0, servo_max)
	pwm.set_pwm(1, 0, servo_max) #Delante.
	pwm.set_pwm(5, 0, servo_min) #Delante.

def stop(pwm):
	pwm.set_pwm(2, 0, 0) #Atras Derecha
	pwm.set_pwm(6, 0, 0) #Atras Izquierda
	pwm.set_pwm(1, 0, 0) #Delante Derecha
	pwm.set_pwm(5, 0, 0) #Delante Izquierda

# print('Moving servo on channel 0, press Ctrl-C to quit...')
#miPwm = iniciarPWM()

if __name__ == "__main__":
	while True:
		if keyboard.is_pressed("e"):
			stop(miPwm)
		if keyboard.is_pressed("q"):
			stop(miPwm)
			break
		if keyboard.is_pressed("w"):
			forward(miPwm)
		if keyboard.is_pressed("a"):
			giroIzquierda(miPwm)
		if keyboard.is_pressed("s"):
			backward(miPwm)
		if keyboard.is_pressed("d"):
			giroDerecha(miPwm)
		giroDerecha(miPwm)
		miPwm.set_pwm(0, 0, servo_max)
		miPwm.set_pwm(4, 0, servo_max)
		# time.sleep(10)
