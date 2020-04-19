import cv2
import numpy as np
from copy import deepcopy

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )

# out = cv2.VideoWriter('VideoDeMuestra.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frameDiagonal = deepcopy(frame)
        lower_green = np.array([40, int(20*1.8), 100])
        upper_green = np.array([80, 230, 140])
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
        mask_green = mask_green/255
        mask_green.astype(bool)
        

        lower_left_triangle = np.tril(mask_green, -1) # Lower triangle of an array
        upper_left_triangle = np.flipud(np.tril(np.flipud(mask_green), 0)) # Upper triangle of an array
        lower_right_triangle = np.fliplr(np.tril(np.fliplr(mask_green), -1)) # Lower triangle of an array
        upper_right_triangle = np.fliplr(np.flipud(np.tril(np.flipud(np.fliplr(mask_green)), 0))) # Upper triangle of an array

        y_up, x_up = np.where(upper_left_triangle == 1)
        y_down, x_down = np.where(lower_right_triangle == 1)

        suficientesPuntos = False
        diagonalNoCruza = False

        if len(x_up) > 1800 and len(x_down) > 2000:
            diagonal_right = np.eye(480,640,0,bool)
            suficientesPuntos = True
            # cv2.imshow('upper_left', upper_left_triangle + diagonal_right)
        else:
            pass
            # cv2.imshow('upper_left', upper_left_triangle)

        
        m_up = 0
        m_down = 0
        b_up = 0
        b_down = 0

        if x_up.size > 50:    
            m_up,b_up = np.polyfit(x_up,y_up,1)

        if x_down.size > 50:    
            m_down,b_down = np.polyfit(x_down,y_down,1)

        b_diag_azul = 0
        b_diag_amarilla = 480
        m_diag_azul = 480/640
        m_diag_amarilla = -480/640

        try:
            # x_azul_contra_up = (b_diag_azul-b_up)/(m_up-m_diag_azul)
            # x_azul_contra_down = (b_diag_azul-b_down)/(m_down-b_diag_azul)
            x_amarilla_contra_up = (b_diag_amarilla-b_up)/(m_up-m_diag_amarilla)
            x_amarilla_contra_down = (b_diag_amarilla-b_down)/(m_down-m_diag_amarilla)

            y_amarilla_contra_up = m_diag_amarilla * x_amarilla_contra_up + b_diag_amarilla
            y_amarilla_contra_down = m_diag_amarilla * x_amarilla_contra_down + b_diag_amarilla


            if not((0 < x_amarilla_contra_up < 640) and (0 < y_amarilla_contra_up < 480)) and not((0 < x_amarilla_contra_down < 640) and (0 < y_amarilla_contra_down < 480)):
                diagonalNoCruza = True
        except:
            pass

        if suficientesPuntos and diagonalNoCruza:
            cv2.line(frameDiagonal,(0,int(b_up)),(int(-b_up/m_up),0),(255,255,255), 2)
            cv2.line(frameDiagonal,(int((480-b_down)/m_down),480),(640,int(640*m_down+b_down)),(255,255,255), 2)
            frame[:,:,2] = frame[:,:,2] + 30

        # cv2.imshow('lower_right', lower_right_triangle)
        cv2.imshow('imagen original', frame)
        # out.write(frameDiagonal)
        # cv2.imshow('eleccion de diagonal', frameDiagonal)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            break
        if key == ord('d') or key == ord('D'):
            print(len(mask_green),len(mask_green[0]))
        if key == ord('c') or key == ord('C'):
            print('cantidad de puntos arriba:', len(x_up))
            print('cantidad de puntos abajo:', len(x_down))

    else:
        break
cap.release()
cv2.destroyAllWindows()