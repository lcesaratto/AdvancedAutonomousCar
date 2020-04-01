from __future__ import division
import time
import Adafruit_PCA9685
import wiringpi as wpi
import keyboard
import sys

def procesoAuxiliar(recibir1):

	class controladorPWM:
		def __init__(self):
			# Initialise the PCA9685 using the default address (0x40).
			self.servo_min = 0  #0  Min pulse length out of 4096
			self.servo_max = 4095  # Max pulse length out of 4096
			self.servo_fw = 0
			self.servo_bw = 0
			self.servo_suave_min = 0
			self.servo_suave_max = 0
			self.servo_brusco_min = 0
			self.servo_brusco_max = 0
			self.pwm = self._iniciarPWM()
			self.tiempoTranscurrido = 0
			self.vehiculoParado = True
			self.tiempoDeAccion = 0

		def _iniciarPWM(self):
			pwm = Adafruit_PCA9685.PCA9685()
			pwm.set_pwm_freq(60)
			return pwm
		
		def setear_parametros(self, servo_fw, servo_bw, servo_suave_min, servo_suave_max, servo_brusco_min, servo_brusco_max, tiempo):
			self.servo_fw = servo_fw
			self.servo_bw = servo_bw
			self.servo_suave_min = servo_suave_min
			self.servo_suave_max = servo_suave_max
			self.servo_brusco_min = servo_brusco_min
			self.servo_brusco_max = servo_brusco_max
			self.tiempoDeAccion = tiempo
			return self

		def start_loop(self):
			# return True
			while True:
				# time.sleep(0.1)
				# if recibir1.poll():
				orden = recibir1.recv()
				print(orden)

				if orden == 'exit':
					sys.exit()

				self.tiempoTranscurrido = time.time()
				self.vehiculoParado = False
				
				# if orden == 'stop':
				# 	self._stop()
				# elif orden == 'forward':
				# 	self._forward()
				# elif orden == 'backward':
				# 	self._backward()
				# elif orden == 'giroBruDer':
				# 	self._giroDerechaBrusco()
				# elif orden == 'giroBruIzq':
				# 	self._giroIzquierdaBrusco()
				# elif orden == 'giroSuaDer':
				# 	self._giroDerechaSuave()
				# elif orden == 'giroSuaIzq':
				# 	self._giroIzquierdaSuave()

				if not self.vehiculoParado and ((time.time() - self.tiempoTranscurrido) > self.tiempoDeAccion):
					print('STOPPING')
					self._stop()

		def _forward(self):
			self.pwm.set_pwm(2, 0, self.servo_min) #Atras Derecha
			self.pwm.set_pwm(6, 0, self.servo_min) #Atras Izquierda
			self.pwm.set_pwm(1, 0, self.servo_fw) #Delante Derecha
			self.pwm.set_pwm(5, 0, self.servo_fw) #Delante Izquierda
			self.pwm.set_pwm(0, 0, self.servo_max)
			self.pwm.set_pwm(4, 0, self.servo_max)

		def _backward(self):
			self.pwm.set_pwm(2, 0, self.servo_bw) #Atras Derecha
			self.pwm.set_pwm(6, 0, self.servo_bw) #Atras Izquierda
			self.pwm.set_pwm(1, 0, self.servo_min) #Delante Derecha
			self.pwm.set_pwm(5, 0, self.servo_min) #Delante Izquierda
			self.pwm.set_pwm(0, 0, self.servo_max)
			self.pwm.set_pwm(4, 0, self.servo_max)

		def _giroDerechaSuave(self):
			self.pwm.set_pwm(2, 0, self.servo_min) #Atras derecha
			self.pwm.set_pwm(6, 0, self.servo_min) #Atras izquierda
			self.pwm.set_pwm(1, 0, self.servo_suave_min) #Delante derecha
			self.pwm.set_pwm(5, 0, self.servo_suave_max) #Delante izquierda
			self.pwm.set_pwm(0, 0, self.servo_max)
			self.pwm.set_pwm(4, 0, self.servo_max)

		def _giroIzquierdaSuave(self):
			self.pwm.set_pwm(2, 0, self.servo_min) #Atras derecha
			self.pwm.set_pwm(6, 0, self.servo_min) #Atras izquierda
			self.pwm.set_pwm(1, 0, self.servo_suave_max) #Delante derecha
			self.pwm.set_pwm(5, 0, self.servo_suave_min) #Delante izquierda
			self.pwm.set_pwm(0, 0, self.servo_max)
			self.pwm.set_pwm(4, 0, self.servo_max)

		def _giroDerechaBrusco(self):
			self.pwm.set_pwm(2, 0, self.servo_brusco_min) #Atras derecha
			self.pwm.set_pwm(6, 0, self.servo_min) #Atras izquierda
			self.pwm.set_pwm(1, 0, self.servo_min) #Delante derecha
			self.pwm.set_pwm(5, 0, self.servo_brusco_max) #Delante izquierda
			self.pwm.set_pwm(0, 0, self.servo_max)
			self.pwm.set_pwm(4, 0, self.servo_max)

		def _giroIzquierdaBrusco(self):
			self.pwm.set_pwm(2, 0, self.servo_min) #Atras derecha
			self.pwm.set_pwm(6, 0, self.servo_brusco_min) #Atras izquierda
			self.pwm.set_pwm(1, 0, self.servo_brusco_max) #Delante derecha
			self.pwm.set_pwm(5, 0, self.servo_min) #Delante izquierda
			self.pwm.set_pwm(0, 0, self.servo_max)
			self.pwm.set_pwm(4, 0, self.servo_max)

		def _stop(self):
			self.vehiculoParado = True
			self.pwm.set_pwm(2, 0, 0) #Atras Derecha
			self.pwm.set_pwm(6, 0, 0) #Atras Izquierda
			self.pwm.set_pwm(1, 0, 0) #Delante Derecha
			self.pwm.set_pwm(5, 0, 0) #Delante Izquierda

	controladorPwm = controladorPWM()
	controladorPwm.setear_parametros(servo_fw=1200, servo_bw=1300, 
                         servo_suave_min=1700, servo_suave_max=500, 
                         servo_brusco_min=1000, servo_brusco_max=2450, tiempo=0.1)
	controladorPwm.start_loop()