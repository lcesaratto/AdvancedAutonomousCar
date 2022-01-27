import cv2
import numpy as np
import copy
import statistics

cap = cv2.VideoCapture(0)
width = 640
multiplicadorLuminosidadAmbiente = 2.5

while cap.isOpened():
    ret, frameOriginal = cap.read()
    if ret:
        frame = frameOriginal[320:480,0:int(width)] #

        lower_green = np.array([40, int(20*multiplicadorLuminosidadAmbiente), 100]) #lower_green = np.array([40, int(20*self.multiplicadorLuminosidadAmbiente), 100])
        # lower_green = np.array([40, 50, 80])
        upper_green = np.array([80, 230, 140])
        
        #Aplico filtro de color con los parametros ya definidos
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
        #kernel = np.ones((3,3), np.uint8)
        #mask_green_a = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        #kernel = np.ones((7,7), np.uint8)
        #mask_green_e = cv2.dilate(mask_green, kernel, iterations=1)
        #kernel = np.ones((11,11), np.uint8)
        #mask_green_c = cv2.morphologyEx(mask_green_e, cv2.MORPH_CLOSE, kernel)
        y, x = np.where(mask_green == 255)
        try:
            if len(x) < 100:
                x_mid = 0
            else:
                x_mid= statistics.median(x)
        except:
            x_mid = 0
        x_mid_int=int(round(x_mid))
        ubicacion_punto_verde = x_mid_int
        # print(self.ubicacion_punto_verde)
        distancia_al_centro = (width/2) - ubicacion_punto_verde
        if x.size > 50 and distancia_al_centro < 320:    
            m,b = np.polyfit(x,y,1)
            print('pendiente:',m)
            if m > 0:
                print('true')
            else:
                print('false')

        cv2.imshow('imagen', mask_green)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            break

cap.release()
# Closes all the frames
cv2.destroyAllWindows()