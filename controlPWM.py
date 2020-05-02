from __future__ import division
import time
import Adafruit_PCA9685
import wiringpi as wpi
import keyboard
import sys

def procesoAuxiliar(recibir1):

	class controladorPWM:
		def __init__(self):
			self.servo_min = 0
			self.servo_max = 4095
			self.servo_fw = 0
			self.servo_bw = 0
			self.servo_suave_min = 0
			self.servo_suave_max = 0
			self.servo_brusco_min = 0
			self.servo_brusco_max = 0
			self.servo_en_el_lugar = 0
			self.pwm = self._iniciarPWM()
			self.ordenDada = 'none'
			self.seDioOrdenConPrioridad = False
			# self.soloOrdenesPrioritariasPermitidas = False

		def _iniciarPWM(self):
			# Initialise the PCA9685 using the default address (0x40).
			pwm = Adafruit_PCA9685.PCA9685()
			pwm.set_pwm_freq(60)
			return pwm

		def _delayPersonalizado(self, tiempo_espera):
			tiempo_inicial = time.time()
			while (time.time()-tiempo_inicial) < tiempo_espera:
				if self.seDioOrdenConPrioridad:
					break
				time.sleep(0.005)
				while recibir1.poll():
					ordenDada = recibir1.recv()
					if (ordenDada == 'exit') or (ordenDada == 'stopAndIgnore') or (ordenDada == 'stopPrioritario'):
						self.seDioOrdenConPrioridad = True
						self.ordenDada = ordenDada

		
		def setear_parametros(self, servo_fw, servo_bw, servo_suave_min, servo_suave_max, 
							  servo_brusco_min, servo_brusco_max, servo_en_el_lugar):
			self.servo_fw = servo_fw
			self.servo_bw = servo_bw
			self.servo_suave_min = servo_suave_min
			self.servo_suave_max = servo_suave_max
			self.servo_brusco_min = servo_brusco_min
			self.servo_brusco_max = servo_brusco_max
			self.servo_en_el_lugar = servo_en_el_lugar
			return self

		def start_loop(self):
			while True:
				if self.seDioOrdenConPrioridad:
					orden = self.ordenDada
					self.seDioOrdenConPrioridad = False
				else:
					orden = recibir1.recv()

				if orden == 'exit':
					self._stop()
					sys.exit()
				# elif orden == 'habilitarOrdenesPrioritarias':
				# 	self._stop()
				# 	self.soloOrdenesPrioritariasPermitidas = True
				# elif orden == 'deshabilitarOrdenesPrioritarias':
				# 	self._stop()
				# 	self.soloOrdenesPrioritariasPermitidas = False
				# 	self._delayPersonalizado(5)
				elif orden == 'stopAndIgnore5s':
					self._stop()
					self._delayPersonalizado(5)
					# time.sleep(5)
				elif orden == 'stopPrioritario':
					self._stop()
				elif orden == 'stop':
					self._stop()
					self._delayPersonalizado(0.4)
				elif orden == 'forward0.5s':
					self._forward()
					self._delayPersonalizado(0.5)
					# time.sleep(0.2)
					self._stop()
					self._delayPersonalizado(0.04)
				elif orden == 'forward':
					self._forward()
					self._delayPersonalizado(0.15)
					# time.sleep(0.2)
					self._stop()
					self._delayPersonalizado(0.04)
				elif orden == 'backward':
					self._backward()
					self._delayPersonalizado(0.2)
					# time.sleep(0.2)
					self._stop()
					self._delayPersonalizado(0.04)
				elif orden == 'giroBruDer':
					self._giroDerechaBrusco()
					self._delayPersonalizado(0.1)
					# time.sleep(0.1)
					self._stop()
					self._delayPersonalizado(0.04)
				elif orden == 'giroBruIzq':
					self._giroIzquierdaBrusco()
					self._delayPersonalizado(0.1)
					# time.sleep(0.1)
					self._stop()
					self._delayPersonalizado(0.04)
				elif orden == 'giroSuaDer':
					self._giroDerechaSuave()
					self._delayPersonalizado(0.1)
					# time.sleep(0.1)
					self._stop()
					self._delayPersonalizado(0.04)
				elif orden == 'giroSuaIzq':
					self._giroIzquierdaSuave()
					self._delayPersonalizado(0.1)
					# time.sleep(0.1)
					self._stop()
					self._delayPersonalizado(0.04)
				elif orden == 'giroEnElLugarIzq':
					self._giroIzquierdaEnElLugar()
					time.sleep(0.1)
					# time.sleep(0.1)
					self._stop()
					self._delayPersonalizado(0.04)
				elif orden == 'giroEnElLugarDer':
					self._giroDerechaEnElLugar()
					time.sleep(0.1)
					# time.sleep(0.1)
					self._stop()
					self._delayPersonalizado(0.04)
				# time.sleep(0.4)
				# while recibir1.poll():
				# 	ordenDada = recibir1.recv()
				# 	if (ordenDada == 'exit') or (ordenDada == 'stopAndIgnore') or (ordenDada == 'stopPrioritario'):
				# 		self.seDioOrdenConPrioridad = True
				# 		self.ordenDada = ordenDada

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
			self.pwm.set_pwm(1, 0, self.servo_suave_max) #Delante derecha
			self.pwm.set_pwm(5, 0, self.servo_suave_min) #Delante izquierda
			self.pwm.set_pwm(0, 0, self.servo_max)
			self.pwm.set_pwm(4, 0, self.servo_max)

		def _giroIzquierdaSuave(self):
			self.pwm.set_pwm(2, 0, self.servo_min) #Atras derecha
			self.pwm.set_pwm(6, 0, self.servo_min) #Atras izquierda
			self.pwm.set_pwm(1, 0, self.servo_suave_min) #Delante derecha
			self.pwm.set_pwm(5, 0, self.servo_suave_max) #Delante izquierda
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

		def _giroIzquierdaEnElLugar(self):
			self.pwm.set_pwm(2, 0, self.servo_min) #Atras derecha
			self.pwm.set_pwm(6, 0, self.servo_brusco_min) #Atras izquierda
			self.pwm.set_pwm(1, 0, self.servo_brusco_max) #Delante derecha
			self.pwm.set_pwm(5, 0, self.servo_min) #Delante izquierda
			self.pwm.set_pwm(0, 0, self.servo_max)
			self.pwm.set_pwm(4, 0, self.servo_max)

		def _giroDerechaEnElLugar(self):
			self.pwm.set_pwm(2, 0, self.servo_brusco_min) #Atras derecha
			self.pwm.set_pwm(6, 0, self.servo_min) #Atras izquierda
			self.pwm.set_pwm(1, 0, self.servo_min) #Delante derecha
			self.pwm.set_pwm(5, 0, self.servo_brusco_max) #Delante izquierda
			self.pwm.set_pwm(0, 0, self.servo_max)
			self.pwm.set_pwm(4, 0, self.servo_max)

		def _stop(self):
			self.pwm.set_pwm(2, 0, 0) #Atras Derecha
			self.pwm.set_pwm(6, 0, 0) #Atras Izquierda
			self.pwm.set_pwm(1, 0, 0) #Delante Derecha
			self.pwm.set_pwm(5, 0, 0) #Delante Izquierda

	controladorPwm = controladorPWM()
	controladorPwm.setear_parametros(servo_fw=1100, servo_bw=1100, 
                         servo_suave_min=2300, servo_suave_max=600, 
                         servo_brusco_min=1000, servo_brusco_max=2600,
						 servo_en_el_lugar = 2000)
	controladorPwm.start_loop()