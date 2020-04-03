import cv2
import numpy as np

cap = cv2.VideoCapture('outputMan2.avi')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
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

        if len(x_up) > 600 and len(x_down) > 1300:
            diagonal_right = np.eye(480,640,0,bool)
            cv2.imshow('upper_left', upper_left_triangle + diagonal_right)
        else:
            cv2.imshow('upper_left', upper_left_triangle)

        cv2.imshow('lower_right', frame)
        key = cv2.waitKey(200)
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