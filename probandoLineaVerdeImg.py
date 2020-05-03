import cv2
import numpy as np
import copy
import statistics

cap = cv2.VideoCapture('imgverde.jpg')
width = 640
i = 1
while cap.isOpened():
    ret, frameOriginal = cap.read()
    if ret:
        frameCamara = copy.deepcopy(frameOriginal)
        frame = copy.deepcopy(frameOriginal[320:480,0:640]) #320,480
        #Defino parametros HSV para detectar color verde 
        lower_green = np.array([40, int(20*2.5), 100])
        upper_green = np.array([80, 230, 140])
        #Aplico filtro de color con los parametros ya definidos
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
        y, x = np.where(mask_green == 255)
        # print(y)
        y += 320
        # cv2.imshow('imagenverde', mask_green)
        print(len(x))
        try:
            if len(x) < 100:
                x_mid = 0
            else:
                x_mid= statistics.median(x)
                y_mid = statistics.median(y)
        except:
            x_mid = 0
        x_mid_int=int(round(x_mid))
        print(x_mid)
        print(y_mid)
        ubicacion_punto_verde = x_mid_int
        distancia_al_centro = 320 - ubicacion_punto_verde
        if x.size > 50 and distancia_al_centro < 320:
            m,b = np.polyfit(y,x,1)
        print(m,b)
        m2 = 1/m
        b2 = -b/m
        print(m2,b2)
        print(int((320-b2)/m2))
        print(int((480-b2)/m2))
        
        # cv2.line(mask_green,(ubicacion_punto_verde,0),(ubicacion_punto_verde,480),(255,255,255), 2)

        frameAMostrar = frameOriginal[320:480,0:640]
        frameAMostrar[:,:,0] = mask_green
        frameAMostrar[:,:,1] = mask_green
        frameAMostrar[:,:,2] = mask_green
        cv2.line(frameAMostrar,(ubicacion_punto_verde,0),(ubicacion_punto_verde,480),(0,255,0), 3)
        cv2.line(frameCamara, (int((320-b2)/m2),320), (int((480-b2)/m2),480), (255,0,0), 3)

        frameCamara[320:480,:,2] = frameCamara[320:640,:,2] + 60

        
        cv2.imshow('imagen', frameCamara)
        # cv2.imshow('original', frameCamara)
        key = cv2.waitKey(10000)
        if key == ord('q') or key == ord('Q'):
            break
        if key == ord('s') or key == ord('S'):
            continue
        if key == ord('c') or key == ord('C'):
            cv2.imwrite("NlineaVerde"+str(i)+".jpg", frameAMostrar)
            cv2.imwrite("NlineaVerdeOriginal"+str(i)+".jpg", frameCamara)
            i += 1
cap.release()
# Closes all the frames
cv2.destroyAllWindows()