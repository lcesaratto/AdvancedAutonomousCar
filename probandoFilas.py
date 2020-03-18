import cv2

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

filasDeseadas = [2,5]

while cap.isOpened():
    ret, frameCompleto = cap.read()
    if ret:

        for fila in filasDeseadas: #Recorre de abajo para arriba, de izquierda a derecha
            for columna in range(16):
                frameOriginalCut=frame[(200-row*40):(240-row*40),(40*column):(40+40*column)]
                frameOriginalCut[:,:,2] = frameOriginalCut[:,:,2] + 30
                for x in range(40): #filas
                    for j in range(40): #columnas
                        frame[200-fila*40+x][0+40*columna+j] = frameOriginalCut[x][j]

        dx, dy = 40,40
        # Custom (rgb) grid color
        grid_color = [255,0,0]# [0,0,0]
        # Modify the image to include the grid
        frame[:,::dy,:] = grid_color
        frame[::dx,:,:] = grid_color

        cv2.imshow('imagen', frame)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            stop(self.miPwm)
            break

    self.cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()