import cv2
import numpy as np
from pyzbar import pyzbar
from MLfunctions import ML_create, ML_test

def line_keeping():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('Videos/20191012_213614.mp4')#('Videos/WhatsApp Video 2019-10-12 at 6.19.29 PM(2).mp4')

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()


        if ret:

            frame = cv2.flip(frame, flipCode=-1)
            #Defino parametros HLS o HSV para detectar solo lineas negras 
            lower_black = np.array([0, 0, 0]) #108 6 17     40 16 37
            upper_black = np.array([150, 75, 255]) #HSV 255, 255, 90 #HLS[150, 75, 255]
            #Divido la imagen en 2 frames distintos (derecha e izquierda) para detectar una sola linea por frame
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
            fps =  cap.get(cv2.CAP_PROP_FPS)
            #frame_right = frame[0:int(height*0.5),0:int(width*0.5)]#frame_right = frame[0:int(height*0.5),0:int(width*0.5)]
            frame_right = frame[0:int(height*0.5),0:int(width*0.4)] #0.5
            frame_left = frame[0:int(height*0.5),int(width*0.5):int(width)]
            frame_left = cv2.flip(frame_left, flipCode=-1)
            frame_right = cv2.flip(frame_right, flipCode=-1)            
            try:
                # ------- Frame Izquierdo -------
                #Aplico filtro de color con los parametros ya definidos
                hsv_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2HLS) #BRG2HLS
                mask_left = cv2.inRange(hsv_left, lower_black, upper_black)
                res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)

                #Aplico filtro pasa bajos y deteccion de lineas por Canny
                dif_gray_left = cv2.GaussianBlur(mask_left, (3, 3), 0) #(mask, (5, 5), 0)
                canny_left = cv2.Canny(dif_gray_left, 1, 500) # 25, 175)
                #Aplico Transformada de Hough
                lines_left = cv2.HoughLinesP(canny_left, 1, np.pi / 180, 20, minLineLength=5, maxLineGap=20) #(canny, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=150)
                # Draw lines on the image
                for line in lines_left: #for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame_left, (x1, y1), (x2, y2), (255, 0, 0), 5)

            except:
                print("No se detectaron lineas izquierda")
            try:
                # ------- Frame Derecho -------
                #Aplico filtro de color con los parametros ya definidos
                hsv_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2HLS) #BRG2HLS
                mask_right = cv2.inRange(hsv_right, lower_black, upper_black)
                res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)

                #Aplico filtro pasa bajos y deteccion de lineas por Canny
                dif_gray_right = cv2.GaussianBlur(mask_right, (3, 3), 0) #(mask, (5, 5), 0)
                canny_right = cv2.Canny(dif_gray_right, 1, 500) # 25, 175)
                #Aplico Transformada de Hough
                lines_right = cv2.HoughLinesP(canny_right, 1, np.pi / 180, 20, minLineLength=5, maxLineGap=10) #(canny, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=150)
                # Draw lines on the image
                cant_lineas=len(lines_right)
                #print(cant_lineas)
                
                if cant_lineas>7:
                    print('Se detecto una curva')
                else:
                    print('Se detecto una linea')
                                
                
                x1prom=0
                x2prom=0
                y1prom=0
                y2prom=0
                cant=0

                righty_sum=0
                lefty_sum=0
                counter=0
                x1m=0;
                for line in lines_right: #for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if(x1>x1m):
                        x1m=x1
                        right_points = [(x1,y1), (x2,y2)]
                #right_points = [(x1,y1), (x2,y2)]
                x1m=0
                [vx,vy,x,y] = cv2.fitLine(np.array(right_points, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Now find two extreme points on the line to draw line
                lefty = int((-x*vy/vx) + y)
                righty = int(((frame_right.shape[1]-x)*vy/vx)+y)

                    #Finally draw the line
                if (abs(vy/vx) > 1) & (abs(vy/vx) < 30) :
                    righty_sum+=righty
                    lefty_sum+=lefty
                    counter+=1
                if (counter!=0) & (righty_sum!=0) & (lefty_sum!=0):
                    righty=righty_sum//counter
                    lefty=lefty_sum//counter
                    cv2.line(frame_right,(frame_right.shape[1]-1,righty),(0,lefty),255,2)
                    print(frame_right.shape[1]-1)
            except Exception as e:
                print(e)
                #print("No se detectaron lineas derecha")            
            # Show result
            #cv2.imshow("Result Image", frame)

            # Display the resulting frame
            #cv2.imshow('Frame3', res_left)
            cv2.imshow('Frame', frame_left)
            cv2.imshow('Frame2', frame_right)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):  # 25fps
                break
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def qr_reader():

    img_path = 'image.jpg'

    img = cv2.imread(img_path)

    barcodes = pyzbar.decode(img)

    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print("[INFO] found {} barcode {}".format(barcodeType, barcodeData))
    cv2.imwrite("new_img.jpg", img)


if __name__ == "__main__":
    line_keeping()
    # qr_reader()
    # ML_test()
