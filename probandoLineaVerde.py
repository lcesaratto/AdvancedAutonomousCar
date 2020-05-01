import cv2
import numpy as np
import copy
import statistics

cap = cv2.VideoCapture(0)
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
        try:
            if len(x) < 100:
                x_mid = 0
            else:
                x_mid= statistics.median(x)
        except:
            x_mid = 0
        x_mid_int=int(round(x_mid))
        ubicacion_punto_verde = x_mid_int
        # cv2.line(mask_green,(ubicacion_punto_verde,0),(ubicacion_punto_verde,480),(255,255,255), 2)

        frameAMostrar = frameOriginal[320:480,0:640]
        frameAMostrar[:,:,0] = mask_green
        frameAMostrar[:,:,1] = mask_green
        frameAMostrar[:,:,2] = mask_green
        cv2.line(frameAMostrar,(ubicacion_punto_verde,0),(ubicacion_punto_verde,480),(0,255,0), 3)

        frameCamara[320:480,:,2] = frameCamara[320:640,:,2] + 60

        
        cv2.imshow('imagen', frameAMostrar)
        # cv2.imshow('original', frameCamara)
        key = cv2.waitKey(1000)
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