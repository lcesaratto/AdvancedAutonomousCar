import cv2

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

filasDeseadas = [1, 10]

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outputMan2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        out.write(frame)

        for row in filasDeseadas: #Recorre de abajo para arriba, de izquierda a derecha
            for column in range(16):
                frameOriginalCut=frame[(440-row*40):(480-row*40),(40*column):(40+40*column)]
                frameOriginalCut[:,:,2] = frameOriginalCut[:,:,2] + 30

                for x in range(40): #filas
                    for j in range(40): #columnas
                        frame[440-row*40+x][0+40*column+j] = frameOriginalCut[x][j]

        dx, dy = 40,40
        # Custom (rgb) grid color
        grid_color = [255,0,0]# [0,0,0]
        # Modify the image to include the grid
        frame[:,::dy,:] = grid_color
        frame[::dx,:,:] = grid_color

        cv2.imshow('imagen', frame)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            break

cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()