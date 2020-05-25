import cv2
import numpy as np
from copy import deepcopy

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )

# out = cv2.VideoWriter('VideoDeMuestra.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))

i = 10

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frameCamara = deepcopy(frame)
        frameOriginal = deepcopy(frame)
        lower_green = np.array([40, int(20*1.8), 100])
        upper_green = np.array([80, 230, 140])
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
        mask_green = mask_green/255
        mask_green.astype(bool)
        for i in range(2):
            suficientesPuntos = False
            diagonalNoCruza = False
            m_up = 0
            m_down = 0
            b_up = 0
            b_down = 0
            puntos_arriba = 1000
            puntos_abajo = 1500
            if i==0:
                # Chequeo diagonal amarilla
                upper_left_triangle = np.flipud(np.tril(np.flipud(mask_green), 0)) # Upper triangle of an array
                lower_right_triangle = np.fliplr(np.tril(np.fliplr(mask_green), -1)) # Lower triangle of an array
                y_up_left, x_up_left = np.where(upper_left_triangle == 1)
                y_down_right, x_down_right = np.where(lower_right_triangle == 1)

                if len(x_up_left) > puntos_arriba and len(x_down_right) > puntos_abajo:
                    # diagonal_right = np.eye(480,640,0,bool)
                    suficientesPuntos = True
                else:
                    continue
                if x_up_left.size > 50:    
                    m_up,b_up = np.polyfit(x_up_left,y_up_left,1)
                if x_down_right.size > 50:    
                    m_down,b_down = np.polyfit(x_down_right,y_down_right,1)
                b_diag_amarilla = 480
                m_diag_amarilla = -480/640
                try:
                    x_amarilla_contra_up = (b_diag_amarilla-b_up)/(m_up-m_diag_amarilla)
                    x_amarilla_contra_down = (b_diag_amarilla-b_down)/(m_down-m_diag_amarilla)
                    y_amarilla_contra_up = m_diag_amarilla * x_amarilla_contra_up + b_diag_amarilla
                    y_amarilla_contra_down = m_diag_amarilla * x_amarilla_contra_down + b_diag_amarilla
                    if not((0 < x_amarilla_contra_up < 640) and (0 < y_amarilla_contra_up < 480)) and not((0 < x_amarilla_contra_down < 640) and (0 < y_amarilla_contra_down < 480)):
                        diagonalNoCruza = True
                    else:
                        continue
                except:
                    continue
                if suficientesPuntos and diagonalNoCruza:
                    print('lala')
                    frameOriginal[:,:,0] =  lower_right_triangle*255 + upper_left_triangle*255 
                    frameOriginal[:,:,1] =  lower_right_triangle*55 + upper_left_triangle*55 + np.flipud(np.tril(np.flipud(np.ones((480,640),int)), 0))*200 + np.fliplr(np.tril(np.fliplr(np.ones((480,640),int)), -1))*200
                    frameOriginal[:,:,2] =  lower_right_triangle*55 + upper_left_triangle*55 + np.flipud(np.tril(np.flipud(np.ones((480,640),int)), 0))*200 + np.fliplr(np.tril(np.fliplr(np.ones((480,640),int)), -1))*200
                    cv2.line(frameOriginal,(0,480),(640,0),(0,255,255), 2)
                    cv2.line(frameOriginal,(0,int(b_up)),(int(-b_up/m_up),0),(0,0,255), 2)
                    cv2.line(frameOriginal,(int((480-b_down)/m_down),480),(640,int(640*m_down+b_down)),(0,0,255), 2)
                    break

            else:
                print('cheqeo espejada')
                # Chequeo diagonal azul
                lower_left_triangle = np.tril(mask_green, -1) # Lower triangle of an array
                upper_right_triangle = np.fliplr(np.flipud(np.tril(np.flipud(np.fliplr(mask_green)), 0))) # Upper triangle of an array
                y_up_right, x_up_right = np.where(upper_right_triangle == 1)
                y_down_left, x_down_left = np.where(lower_left_triangle == 1)

                if len(x_up_right) > puntos_arriba and len(x_down_left) > puntos_abajo:
                    suficientesPuntos = True
                    print('primera condicion ok')
                else:
                    print('no,', len(x_up_right), len(x_down_left))
                    break
                if x_up_right.size > 50:    
                    m_up,b_up = np.polyfit(x_up_right,y_up_right,1)
                if x_down_left.size > 50:    
                    m_down,b_down = np.polyfit(x_down_left,y_down_left,1)
                b_diag_azul = 0
                m_diag_azul = 480/640
                try:
                    x_azul_contra_up = (b_diag_azul-b_up)/(m_up-m_diag_azul)
                    x_azul_contra_down = (b_diag_azul-b_down)/(m_down-b_diag_azul)
                    y_azul_contra_up = m_diag_azul * x_azul_contra_up + b_diag_azul
                    y_azul_contra_down = m_diag_azul * x_azul_contra_down + b_diag_azul
                    if not((0 < x_azul_contra_up < 640) and (0 < y_azul_contra_up < 480)) and not((0 < x_azul_contra_down < 640) and (0 < y_azul_contra_down < 480)):
                        diagonalNoCruza = True
                        print('segunda ok')
                except:
                    diagonalNoCruza = False

                if suficientesPuntos and diagonalNoCruza:  
                    print('lala')
                    frameOriginal[:,:,0] =  lower_left_triangle*255 + upper_right_triangle*255
                    frameOriginal[:,:,1] =  lower_left_triangle*55 + upper_right_triangle*55 + np.tril(np.ones((480,640),int), -1)*200 + np.fliplr(np.flipud(np.tril(np.flipud(np.fliplr(np.ones((480,640),int))), 0)))*200
                    frameOriginal[:,:,2] =  lower_left_triangle*55 + upper_right_triangle*55 + np.tril(np.ones((480,640),int), -1)*200 + np.fliplr(np.flipud(np.tril(np.flipud(np.fliplr(np.ones((480,640),int))), 0)))*200
                    cv2.line(frameOriginal,(0,0),(640,480),(0,255,255), 2)
                    cv2.line(frameOriginal,(0,int(b_up)),(640,int(m_up*640+b_up)),(0,0,255), 2)
                    cv2.line(frameOriginal,(0,int(b_down)),(int((480-b_down)/m_down),480),(0,0,255), 2)


        # cv2.imshow('lower_right', lower_right_triangle)
        cv2.imshow('imagen original', frameOriginal)
        # out.write(frameDiagonal)
        # cv2.imshow('eleccion de diagonal', frameDiagonal)
        key = cv2.waitKey(100)
        if key == ord('q') or key == ord('Q'):
            break
        if key == ord('s') or key == ord('S'):
            continue
        if key == ord('c') or key == ord('C'):
            cv2.imwrite("NprobandoBocacalle"+str(i)+".jpg", frameOriginal)
            cv2.imwrite("NprobandoBocacalleOriginal"+str(i)+".jpg", frameCamara)
            i += 1
        # if key == ord('c') or key == ord('C'):
        #     print('cantidad de puntos arriba:', len(x_up_left))
        #     print('cantidad de puntos abajo:', len(x_down_right))
        # if key == ord('v') or key == ord('V'):
        #     print('cantidad de puntos arriba:', len(x_up_right))
        #     print('cantidad de puntos abajo:', len(x_down_left))

    else:
        break
cap.release()
cv2.destroyAllWindows()