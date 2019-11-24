import cv2
import numpy as np
from pyzbar import pyzbar
from MLfunctions import ML_create, ML_test
import matplotlib.pyplot as plt
# try:
#     from PIL import Image
# except ImportError:
#     import Image

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
                    x1m=0
                    for line in lines_right: #for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if(x1>x1m):
                            x1m=x1
                            right_points = [(x1,y1), (x2,y2)]
                    #right_points = [(x1,y1), (x2,y2)]
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

            # Display the resulting frame
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

def line_keeping_grid_v1():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('Videos/20191012_213614.mp4')#('Videos/WhatsApp Video 2019-10-12 at 6.19.29 PM(2).mp4')

    lower_black = np.array([0, 0, 0]) #108 6 17     40 16 37
    upper_black = np.array([150, 75, 255]) #HSV 255, 255, 90 #HLS[150, 75, 255]

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
            # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
            # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
            fps =  cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
            frame=cv2.flip(frame, flipCode=-1)
            frame = frame[0:int(height*0.5),0:int(width)]


            # print(int(width))    #640 --->32
            # print(int(height*0.5))   #240 --->12
            # 32*12= 384 bloques
            column=0
            row=0
            frameResulting=frame

            #Aplico filtro de color con los parametros ya definidos
            # hsv_right = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS) #BRG2HLS
            hsv_right = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #BRG2HLS
            # cv2.imshow('framefilterHSV', hsv_right)
            mask_right = cv2.inRange(frame, lower_black, upper_black)
            res_right = cv2.bitwise_and(frame, frame, mask=mask_right)
            # cv2.imshow('framefilter', res_right)

            #Aplico filtro pasa bajos y deteccion de lineas por Canny
            '''
            kernel = np.ones((3,3), np.uint8)
            frame_e = cv2.erode(frame, kernel, iterations=1)
            gray_right = cv2.cvtColor(frame_e, cv2.COLOR_BGR2GRAY) 
            (thresh, bw_right) = cv2.threshold(gray_right, 40, 255, cv2.THRESH_BINARY)
            #dif_gray_right = cv2.GaussianBlur(bw_right, (1, 1), 0) #(mask, (5, 5), 0)
            cv2.imshow('framefilterdif', frame_e)
            canny_right = cv2.Canny(bw_right, 25, 175) # 25, 175)
            cv2.imshow('framefilter', canny_right)
            '''
            #Umbral Dinamico
            kernel = np.ones((3,3), np.uint8)
            frame_e = cv2.erode(frame, kernel, iterations=1)
            gray = cv2.cvtColor(frame_e, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.medianBlur(gray, 5)
            dst2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
            cv2.imshow('framefilterdif', dst2)
            dif_gray_right = cv2.GaussianBlur(dst2, (11, 11), 0) #(mask, (5, 5), 0)
            #cv2.imshow('framefilterdif', frame_e)
            canny_right = cv2.Canny(dif_gray_right, 25, 175) # 25, 175)
            kernel = np.ones((7,7), np.uint8)
            canny_right = cv2.morphologyEx(canny_right, cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((3,3), np.uint8)
            canny_right = cv2.erode(canny_right, kernel, iterations=1)
            cv2.imshow('framefilter', canny_right)
            
            # list = []
            '''Recorre de abajo para arriba, de izquierda a derecha'''
            while row < 6:
                if column < 16:

                    cannyCut=canny_right[(200-row*40):(240-row*40),(40*column):(40+40*column)]
                    frameCut=frame[(200-row*40):(240-row*40),(40*column):(40+40*column)]


                    '''Aca procesamos cada frameCut'''
                    try:

                        # frameCut = canny_right
                        ## Aplico Transformada de Hough
                        lines_right = cv2.HoughLinesP(cannyCut, 1, np.pi / 180, 10, minLineLength=0, maxLineGap=1) #(canny, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=150)
                        # Draw lines on the image
                        if lines_right is not None:
                            cant_lineas=len(lines_right)
                            print(cant_lineas)
                            
                            if cant_lineas>100: #11
                                #print('Se detecto una curva')
                                frameCut[:,:,2] = frameCut[:,:,2] + 50
                                #frameCut[:,:,0] = 0
                            else:
                                #print('Se detecto una linea')
                            
                                x1prom=0
                                x2prom=0
                                y1prom=0
                                y2prom=0
                                cant=0

                                righty_sum=0
                                lefty_sum=0
                                counter=0
                                x1m=0
                                for line in lines_right: #for line in lines:
                                    x1, y1, x2, y2 = line[0]
                                    '''
                                    if(x1>x1m):
                                        x1m=x1
                                        right_points = [(x1,y1), (x2,y2)]
                                        '''
                                    #x1m=x1
                                    #right_points = [(x1,y1), (x2,y2)]  
                                    cv2.line(frameCut,(x1,y1),(x2,y2),(0,0,255),2)
                                    #cv2.line(frameCut,(frameCut.shape[1]-1,x1),(0,y1),255,2)
                                '''
                                [vx,vy,x,y] = cv2.fitLine(np.array(right_points, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                                
                                # Now find two extreme points on the line to draw line
                                lefty = int((-x*vy/vx) + y)
                                righty = int(((frameCut.shape[1]-x)*vy/vx)+y)

                                    #Finally draw the line
                                
                                if (abs(vy/vx) > 1) & (abs(vy/vx) < 30) :
                                    righty_sum+=righty
                                    lefty_sum+=lefty
                                    counter+=1
                                '''
                                '''
                                if (counter!=0) & (righty_sum!=0) & (lefty_sum!=0):
                                    righty=righty_sum//counter
                                    lefty=lefty_sum//counter
                                    cv2.line(frameCut,(frameCut.shape[1]-1,righty),(0,lefty),255,2)
                                
                                cv2.line(frameCut,(frameCut.shape[1]-1,righty),(0,lefty),255,2)
                                    # print(frameCut.shape[1]-1)
                                    '''
                                frameCut[:,:,0] = frameCut[:,:,0]+50
                        else:
                            frameCut[:,:,1] = frameCut[:,:,1] + 50        

                    except Exception as e:
                        print(e)

                    '''Aca terminamos de procesar cada frameCut'''

                    for x in range(40): #filas
                        for j in range(40): #columnas
                            # print(frameCut[x][j])
                            frameResulting[200-row*40+x][0+40*column+j] = frameCut[x][j]
                    column+=1
                    # list.append(frameCut)
                elif column == 16:
                    column=0
                    row+=1
            '''     
            if (frame == frameResulting).all(): #Esto es para verificar que el frame original sea igual al reconstruido
                print ("Excelente!")
                '''
            # frameCut=frame[(1):(2),(1):(2)]
            # print("hola0", frame[1][1])
            # print("hola00", frame[239][639])
            # print("hola1", frameCut[0][0])
            # # print("hola2", list[383])
            # cv2.imshow('frameCut', frameCut)

            # Grid lines at these intervals (in pixels)
            # dx and dy can be different
            dx, dy = 40,40
            # Custom (rgb) grid color
            grid_color = [255,0,0]# [0,0,0]
            # Modify the image to include the grid
            frameResulting[:,::dy,:] = grid_color
            frameResulting[::dx,:,:] = grid_color

            cv2.line(frameResulting,(0,140),(640,140),(255,255,255),2)
            # Display the resulting frame
            cv2.imshow('frameResulting', frameResulting)

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

def line_keeping_grid_v2():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    cap = cv2.VideoCapture('Videos/20191012_213614.mp4')#('Videos/WhatsApp Video 2019-10-12 at 6.19.29 PM(2).mp4')

    lower_black = np.array([0, 0, 0]) #108 6 17     40 16 37
    upper_black = np.array([150, 75, 255]) #HSV 255, 255, 90 #HLS[150, 75, 255]

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            right_points_up = np.array([0, 0])
            left_points_up = np.array([0, 0])

            frame = cv2.flip(frame, flipCode=-1)
            #Defino parametros HLS o HSV para detectar solo lineas negras 
            lower_black = np.array([0, 0, 0]) #108 6 17     40 16 37
            upper_black = np.array([150, 75, 255]) #HSV 255, 255, 90 #HLS[150, 75, 255]
            #Divido la imagen en 2 frames distintos (derecha e izquierda) para detectar una sola linea por frame
            # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
            # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
            fps =  cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
            frame=cv2.flip(frame, flipCode=-1)
            frame = frame[0:int(height*0.5),0:int(width)]


            # print(int(width))    #640 --->32
            # print(int(height*0.5))   #240 --->12
            # 32*12= 384 bloques
            column=0
            row=0
            frameResulting=frame

            #Aplico filtro de color con los parametros ya definidos
            # hsv_right = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS) #BRG2HLS
            hsv_right = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #BRG2HLS
            # cv2.imshow('framefilterHSV', hsv_right)
            mask_right = cv2.inRange(frame, lower_black, upper_black)
            res_right = cv2.bitwise_and(frame, frame, mask=mask_right)
            # cv2.imshow('framefilter', res_right)

            #Aplico filtro pasa bajos y deteccion de lineas por Canny
            '''
            kernel = np.ones((3,3), np.uint8)
            frame_e = cv2.erode(frame, kernel, iterations=1)
            gray_right = cv2.cvtColor(frame_e, cv2.COLOR_BGR2GRAY) 
            (thresh, bw_right) = cv2.threshold(gray_right, 40, 255, cv2.THRESH_BINARY)
            #dif_gray_right = cv2.GaussianBlur(bw_right, (1, 1), 0) #(mask, (5, 5), 0)
            cv2.imshow('framefilterdif', frame_e)
            canny_right = cv2.Canny(bw_right, 25, 175) # 25, 175)
            cv2.imshow('framefilter', canny_right)
            '''
            #Umbral Dinamico
            kernel = np.ones((3,3), np.uint8)
            frame_e = cv2.erode(frame, kernel, iterations=1)
            gray = cv2.cvtColor(frame_e, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.medianBlur(gray, 5)
            dst2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
            cv2.imshow('framefilterdif', dst2)
            dif_gray_right = cv2.GaussianBlur(dst2, (11, 11), 0) #(mask, (5, 5), 0)
            #cv2.imshow('framefilterdif', frame_e)
            canny_right = cv2.Canny(dif_gray_right, 25, 175) # 25, 175)
            kernel = np.ones((7,7), np.uint8)
            canny_right = cv2.morphologyEx(canny_right, cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((3,3), np.uint8)
            canny_right = cv2.erode(canny_right, kernel, iterations=1)
            cv2.imshow('framefilter', canny_right)
            
            # list = []
            '''Recorre de abajo para arriba, de izquierda a derecha'''
            while row < 6:
                if column < 16:

                    cannyCut=canny_right[(200-row*40):(240-row*40),(40*column):(40+40*column)]
                    frameCut=frame[(200-row*40):(240-row*40),(40*column):(40+40*column)]

                    if row == 2:
                        '''Aca procesamos cada frameCut'''
                        try:

                            # frameCut = canny_right
                            ## Aplico Transformada de Hough
                            lines_right = cv2.HoughLinesP(cannyCut, 1, np.pi / 180, 10, minLineLength=0, maxLineGap=1) #(canny, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=150)
                            # Draw lines on the image
                            if lines_right is not None:
                                cant_lineas=len(lines_right)
                                #print(cant_lineas)
                                
                                if cant_lineas>100: #11
                                    #print('Se detecto una curva')
                                    frameCut[:,:,2] = frameCut[:,:,2] + 50
                                    #frameCut[:,:,0] = 0
                                else:
                                    #print('Se detecto una linea')
                                
                                    x1prom=0
                                    x2prom=0
                                    y1prom=0
                                    y2prom=0
                                    cant=0

                                    righty_sum=0
                                    lefty_sum=0
                                    counter=0
                                    x1m=640
                                    x1M=0
                                    columnm=16
                                    for line in lines_right: #for line in lines:
                                        x1, y1, x2, y2 = line[0]
                                        
                                        if(x1>x1M):
                                            x1M=x1
                                            right_points_up = [x1+40*column, y1+40*(5-row)]
                                            right_points_down = [x2, y2]
                                        if(column<7):
                                            # print(x1+40*column)
                                            left_points_up = [x1+40*column, y1+40*(5-row)]
                                            left_points_down = [x2, y2]
                                        #x1m=x1
                                        #right_points = [(x1,y1), (x2,y2)]  
                                        cv2.line(frameCut,(x1,y1),(x2,y2),(0,0,255),2)
                                        #cv2.line(frameCut,(frameCut.shape[1]-1,x1),(0,y1),255,2)
                                    '''
                                    [vx,vy,x,y] = cv2.fitLine(np.array(right_points, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                                    
                                    # Now find two extreme points on the line to draw line
                                    lefty = int((-x*vy/vx) + y)
                                    righty = int(((frameCut.shape[1]-x)*vy/vx)+y)

                                        #Finally draw the line
                                    
                                    if (abs(vy/vx) > 1) & (abs(vy/vx) < 30) :
                                        righty_sum+=righty
                                        lefty_sum+=lefty
                                        counter+=1
                                    '''
                                    '''
                                    if (counter!=0) & (righty_sum!=0) & (lefty_sum!=0):
                                        righty=righty_sum//counter
                                        lefty=lefty_sum//counter
                                        cv2.line(frameCut,(frameCut.shape[1]-1,righty),(0,lefty),255,2)
                                    
                                    cv2.line(frameCut,(frameCut.shape[1]-1,righty),(0,lefty),255,2)
                                        # print(frameCut.shape[1]-1)
                                        '''
                                    frameCut[:,:,0] = frameCut[:,:,0]+50
                            else:
                                frameCut[:,:,1] = frameCut[:,:,1] + 50        

                        except Exception as e:
                            print(e)

                        '''Aca terminamos de procesar cada frameCut'''

                    for x in range(40): #filas
                        for j in range(40): #columnas
                            # print(frameCut[x][j])
                            frameResulting[200-row*40+x][0+40*column+j] = frameCut[x][j]
                    column+=1
                    # list.append(frameCut)
                elif column == 16:
                    column=0
                    row+=1
            '''     
            if (frame == frameResulting).all(): #Esto es para verificar que el frame original sea igual al reconstruido
                print ("Excelente!")
                '''
            # frameCut=frame[(1):(2),(1):(2)]
            # print("hola0", frame[1][1])
            # print("hola00", frame[239][639])
            # print("hola1", frameCut[0][0])
            # # print("hola2", list[383])
            # cv2.imshow('frameCut', frameCut)

            # Grid lines at these intervals (in pixels)
            # dx and dy can be different
            dx, dy = 40,40
            # Custom (rgb) grid color
            grid_color = [255,0,0]# [0,0,0]
            # Modify the image to include the grid
            frameResulting[:,::dy,:] = grid_color
            frameResulting[::dx,:,:] = grid_color

            #cv2.line(frameResulting,(0,140),(640,140),(255,255,255),2)
            cv2.line(frameResulting,(left_points_up[0],140),(right_points_up[0],140),(255,255,255),2)
            # Display the resulting frame
            cv2.imshow('frameResulting', frameResulting)

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

if __name__ == "__main__":
    # line_keeping()
    # qr_reader()
    # ML_test()
    line_keeping_grid_v2()