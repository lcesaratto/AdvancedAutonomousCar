from __future__ import division
import time
import Adafruit_PCA9685
import wiringpi as wpi
import keyboard
import sys
import cv2
import numpy as np
import copy
import statistics

cap = cv2.VideoCapture(0)
width = 640


pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)

while cap.isOpened():
    ret, frameOriginal = cap.read()
    if ret:
        #Defino parametros HSV para detectar color rojo 
        frame = copy.deepcopy(frameOriginal)
        lower_red = np.array([0, 10, 40])
        upper_red = np.array([10, 100, 100])
        #Aplico filtro de color con los parametros ya definidos
        hsv_red = cv2.cvtColor(frameOriginal, cv2.COLOR_BGR2HLS)
        mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
        #Proceso
        # if np.mean(mask_red) > 15:
        #     return True
        y, x = np.where(mask_red == 255)
        if (len(x)) > 1000:
            mediana_y = int(statistics.median_low(y))
            cv2.line(mask_red, (0,mediana_y), (640,mediana_y), (255,255,255), 3)
        
        cv2.imshow('imagen', mask_red)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            break
        if key == ord('c'):
            # cv2.imwrite('sendaroja.jpg', frame)
            print(mediana_y)
            print(len(np.where(mask_red == 255)[0]))
        if key == ord('m') or key == ord('M'):
            pwm.set_pwm(2, 0, 0) #Atras Derecha
            pwm.set_pwm(6, 0, 0) #Atras Izquierda
            pwm.set_pwm(1, 0, 1100) #Delante Derecha
            pwm.set_pwm(5, 0, 1100) #Delante Izquierda
            pwm.set_pwm(0, 0, 4095)
            pwm.set_pwm(4, 0, 4095)
            time.sleep(0.5)
            pwm.set_pwm(2, 0, 0) #Atras Derecha
            pwm.set_pwm(6, 0, 0) #Atras Izquierda
            pwm.set_pwm(1, 0, 0) #Delante Derecha
            pwm.set_pwm(5, 0, 0) #Delante Izquierda

cap.release()
# Closes all the frames
cv2.destroyAllWindows()