import cv2
import numpy as np
import copy
import statistics

cap = cv2.VideoCapture(0)
width = 640
i = 1

x_abajo=np.zeros(0)
m_vista=np.zeros(0)
x_medio=np.zeros(0)

y_corr = np.loadtxt('data/m_vista1.out')
x_corr = np.loadtxt('data/x_abajo1.out')
m_corr,b_corr = np.polyfit(x_corr,y_corr,1)

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
        distancia_al_centro = 320 - ubicacion_punto_verde
        if x.size > 50 and distancia_al_centro < 320:
            y+=320    
            m,b = np.polyfit(y,x,1)
            x_ab = m*480+b

            m_tabla = x_ab*m_corr+b_corr
            if (0.85*abs(m_tabla) < abs(m) < 1.15*abs(m_tabla)):
                m2 = 1/m
                b2 = -b/m
                cv2.line(frameCamara, (int((320-b2)/m2),320), (int((480-b2)/m2),480), (255,0,0), 4)
                print('OK')
            else:
                if m < 0:
                    print("GIRAR A LA DERECHA")
                else:
                    print("GIRAR A LA IZQUIERDA")


            # m2 = 1/m
            # b2 = -b/m
            # print(m,b)
            # print('la posicion en x abajo: ', x_ab, 'y la pendiente es: ', m, 'y la posicion en x media: ', x_mid)
            # cv2.line(frameCamara, (int((320-b2)/m2),320), (int((480-b2)/m2),480), (255,0,0), 3)
            
        
        # cv2.line(mask_green,(ubicacion_punto_verde,0),(ubicacion_punto_verde,480),(255,255,255), 2)

        frameAMostrar = frameOriginal[320:480,0:640]
        frameAMostrar[:,:,0] = mask_green
        frameAMostrar[:,:,1] = mask_green
        frameAMostrar[:,:,2] = mask_green
        cv2.line(frameAMostrar,(ubicacion_punto_verde,0),(ubicacion_punto_verde,480),(0,255,0), 3)
        

        frameCamara[320:480,:,2] = frameCamara[320:640,:,2] + 60

        
        cv2.imshow('imagen', frameCamara)
        # cv2.imshow('original', frameCamara)
        key = cv2.waitKey(100)
        if key == ord('q') or key == ord('Q'):
            np.savetxt('x_abajo4.out', x_abajo, delimiter=',')
            np.savetxt('m_vista4.out', m_vista, delimiter=',')
            np.savetxt('x_medio4.out', x_medio, delimiter=',')
            break
        if key == ord('s') or key == ord('S'):
            print(x_abajo,m_vista,x_medio)
        if key == ord('c') or key == ord('C'):
            # cv2.imwrite("NlineaVerde"+str(i)+".jpg", frameAMostrar)
            # cv2.imwrite("NlineaVerdeOriginal"+str(i)+".jpg", frameCamara)
            # i += 1
            x_abajo = np.append(x_abajo,x_ab)
            m_vista = np.append(m_vista,m)
            x_medio = np.append(x_medio, x_mid)
            print(x_abajo,m_vista,x_medio)
            np.savetxt('x_abajo4.out', x_abajo, delimiter=',')
            np.savetxt('m_vista4.out', m_vista, delimiter=',')
            np.savetxt('x_medio4.out', x_medio, delimiter=',')

cap.release()
# Closes all the frames
cv2.destroyAllWindows()