import cv2
import numpy as np
import copy
import statistics

cap = cv2.VideoCapture(0)
width = 640

while cap.isOpened():
    ret, frameOriginal = cap.read()
    if ret:
        #Defino parametros HSV para detectar color rojo 
        frame = copy.deepcopy(frameOriginal)
        lower_red = np.array([170, 179, 0])
        upper_red = np.array([255, 255, 255])
        #Aplico filtro de color con los parametros ya definidos
        hsv_red = cv2.cvtColor(frameOriginal, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
        #Proceso
        # if np.mean(mask_red) > 15:
        #     return True
        
        cv2.imshow('imagen', mask_red)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            break
        if key == ord('c'):
            cv2.imwrite('sendaroja.jpg', frame)
            print(len(np.where(mask_red == 255)[0]))
cap.release()
# Closes all the frames
cv2.destroyAllWindows()