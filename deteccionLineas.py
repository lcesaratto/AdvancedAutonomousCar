import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
from detectorQR import qr_reader

class SeguimientoLineas (object):
    def __init__(self):
        self.cap = self._abrirCamara()
        self.fps, self.width, self.height = self._obtenerParametrosFrame()
        # Frame con el que trabajan todos los metodos
        self.frameProcesado = []
        # Especificamos mediante los siguientes array que porciones de frame procesar
        self.filasDeseadas = [2]
        self.columnasDeseadas = []
        # Variables y banderas utulizadas cuando se detecta la curva derecha
        self.dentroCurvaDerecha = False
        self.distanciaDerecha_arr = np.zeros(10)
        self.distanciaDerecha_med = 0
        self.indice_ultima_posicion_3 = 0
        self.array3_listo = False
        self.last = 0
        self.up_point = np.array([0,0])
        # Variables y banderas utilizadas cuando se detecta la bocacalle
        self.bocacalleDetectada=False
        self.right_points_up_arr = np.zeros(10)
        self.left_points_up_arr = np.zeros(10)
        self.right_points_up_med = 0
        self.left_points_up_med = 0
        self.left_points_up_last = np.array([0, 0])
        self.right_points_up_last = np.array([0, 0])
        self.dentroDeBocacalle = False
        self.ultimas_posiciones = np.zeros(10) # Array para almacenar las ultimas 10 posiciones del vehiculo
        self.indice_ultima_posicion = 0
        self.indice_ultima_posicion_2 = 0
        # Cuando se detecta el cartel o se pulsa la letra K
        self.cartelDetectado = False
        # Ubicacion de los puntos a las lineas laterales
        self.right_points_up = np.array([0, 0]) # Para la fila 2
        self.left_points_up = np.array([0, 0])
        self.right_points_up_2 = np.array([0, 0]) # Para la fila 5
        self.left_points_up_2 = np.array([0, 0])
        # Array que determina la posicion hacia donde ir
        self.accionATomar = [0, 0, 0, 0]


        # Banderas de prueba
        self.depositoDetectado = 3
        self.depositoABuscar = 3
        self.activarBuscarFramesLineaPunteada = False
        
    def _abrirCamara (self):
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        #cap = cv2.VideoCapture('Videos/20200107_163552.mp4')
        cap = cv2.VideoCapture('Videos/20191012_213614.mp4')#('Videos/WhatsApp Video 2019-10-12 at 6.19.29 PM(2).mp4')
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
        return cap

    def _obtenerParametrosFrame(self):
        fps =  self.cap.get(cv2.CAP_PROP_FPS)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return fps, width, height

    def _prepararFrame (self, frame):
        # frame = cv2.flip(frame, flipCode=-1)
        frame = frame[0:int(self.height*0.5),0:int(self.width)]
        return frame

    def _aplicarFiltrosMascaras (self, frame):
        #Defino parametros HLS o HSV para detectar solo lineas negras 
        lower_black = np.array([0, 0, 0]) #108 6 17     40 16 37
        upper_black = np.array([150, 75, 255]) #HSV 255, 255, 90 #HLS[150, 75, 255]

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
        # cv2.imshow('framefilterdif', dst2)
        dif_gray_right = cv2.GaussianBlur(dst2, (11, 11), 0) #(mask, (5, 5), 0)
        #cv2.imshow('framefilterdif', frame_e)
        canny_right = cv2.Canny(dif_gray_right, 25, 175) # 25, 175)
        kernel = np.ones((7,7), np.uint8)
        canny_right = cv2.morphologyEx(canny_right, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((3,3), np.uint8)
        canny_right = cv2.erode(canny_right, kernel, iterations=1)
        # cv2.imshow('framefilter', canny_right)
        return canny_right

    def _obtenerPorcionesFrame (self, frameOriginal, frame, row, column):
        frameCut=frame[(200-row*40):(240-row*40),(40*column):(40+40*column)]
        frameOriginalCut=frameOriginal[(200-row*40):(240-row*40),(40*column):(40+40*column)]
        return frameCut, frameOriginalCut

    def _procesarPorcionFrame (self, cannyCut, frameCut, fila, columna):

        try:
            # frameCut = canny_right
            ## Aplico Transformada de Hough
            lines = cv2.HoughLinesP(cannyCut, 1, np.pi / 180, 10, minLineLength=0, maxLineGap=1) #(canny, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=150)

            # Draw lines on the image
            if lines is not None:
                cant_lineas=len(lines)
                # print(cant_lineas)
                
                if cant_lineas>100: # print('Se detecto una curva')
                    frameCut[:,:,2] = frameCut[:,:,2] + 30
                else: # print('Se detecto una linea')
                    x1M=0
                    x1M_2=0
                    y1M=0


                    for line in lines:
                        x1, y1, x2, y2 = line[0]

                        if(x1>x1M and fila==2):
                            x1M=x1
                            self.right_points_up = [x1+40*columna, y1+40*(5-fila)]
                            self.right_points_down = [x2, y2]
                        if(columna<7 and fila==2):
                            # print(x1+40*column)
                            self.left_points_up = [x1+40*columna, y1+40*(5-fila)]
                            self.left_points_down = [x2, y2]
                        if(x1>x1M_2 and fila==5):
                            x1M_2=x1
                            self.right_points_up_2 = [x1+40*columna, y1+40*(5-fila)]
                            self.right_points_down_2 = [x2, y2]
                        if(columna<7 and fila==5):
                            # print(x1+40*column)
                            self.left_points_up_2 = [x1+40*columna, y1+40*(5-fila)]
                            self.left_points_down_2 = [x2, y2] 

                        # Guardamos la posicion del punto superior (frente al vehiculo)
                        if self.dentroCurvaDerecha:
                            if (columna is int(self.last/40)) and y1>y1M:
                            # if (self.last*0.7< (x1+40*columna) < self.last*1.3) and y1>y1M:
                                y1M=y1
                                self.up_point = [self.last,y1+40*(5-fila)]
                               
                        # x1m=x1
                        # right_points = [(x1,y1), (x2,y2)]

                        cv2.line(frameCut,(x1,y1),(x2,y2),(0,0,255),2)
                        # cv2.line(frameCut,(frameCut.shape[1]-1,x1),(0,y1),255,2)

                    frameCut[:,:,0] = frameCut[:,:,0]+30
            else:
                frameCut[:,:,1] = frameCut[:,:,1] + 30        

        except Exception as e:
            print(e)
        return frameCut

    def _reconstruirFrame(self, porcionFrameProcesado, fila, columna):
        for x in range(40): #filas
            for j in range(40): #columnas
                self.frameProcesado[200-fila*40+x][0+40*columna+j] = porcionFrameProcesado[x][j]

    def _detectarBocacalle(self):
        if self.bocacalleDetectada:
            cv2.line(self.frameProcesado,(self.left_points_up_last,140),(self.right_points_up_last,140),(0,255,0),2)
        else:
            cv2.line(self.frameProcesado,(self.left_points_up[0],140),(self.right_points_up[0],140),(255,255,255),2)

        cv2.line(self.frameProcesado,(self.left_points_up_2[0],20),(self.right_points_up_2[0],20),(255,255,255),2)
        dist_line_down = self.right_points_up[0] - self.left_points_up[0]
        dist_line_up = self.right_points_up_2[0] - self.left_points_up_2[0]
            
        
        if ((dist_line_down > 200)): #En promedio 170 # or dist_line_down < 150
            if not self.bocacalleDetectada:
                self.right_points_up_last = self.right_points_up_med#int(statistics.median(self.right_points_up_arr))
                self.left_points_up_last = self.left_points_up_med#int(statistics.median(self.left_points_up_arr))
            self.bocacalleDetectada=True
            if dist_line_up < 200:
                self.right_points_up_last = self.right_points_up_2[0]
                self.left_points_up_last = self.left_points_up_2[0]
        else:
            if (self.indice_ultima_posicion_2 is 10):
                self.indice_ultima_posicion_2 = 0
            self.right_points_up_arr[self.indice_ultima_posicion_2] = self.right_points_up[0]
            self.left_points_up_arr[self.indice_ultima_posicion_2] = self.left_points_up[0]
            self.right_points_up_med = int(statistics.median(self.right_points_up_arr))
            self.left_points_up_med = int(statistics.median(self.left_points_up_arr))
            self.indice_ultima_posicion_2 += 1
            if ((self.right_points_up_med*0.9 < self.right_points_up[0] < self.right_points_up_med*1.1) and (self.left_points_up_med*0.9 < self.left_points_up[0] < self.left_points_up_med*1.1)):
                self.bocacalleDetectada=False

    def _detectarCurvaDerecha(self):
        #print(self.dentroCurvaDerecha)
        distanciaDerecha = self.right_points_up[0] - (self.width/2)

        if (self.indice_ultima_posicion_3 is 10):
            self.array3_listo = True
            self.indice_ultima_posicion_3 = 0
        self.distanciaDerecha_arr[self.indice_ultima_posicion_3] = distanciaDerecha
        self.indice_ultima_posicion_3 += 1

        self.distanciaDerecha_med = int(statistics.median(self.distanciaDerecha_arr))

        # cv2.line(self.frameProcesado,(int(distanciaDerecha+320),0),(int(distanciaDerecha+320),240),(0,0,255),2)

        if distanciaDerecha > self.distanciaDerecha_med*1.3 and self.array3_listo == True: # 130: # Condicion de entrada a la curva (INSTANTANEA)
            self.dentroCurvaDerecha = True
            self.last = int(statistics.median(self.ultimas_posiciones))

        elif self.distanciaDerecha_med*0.8 < distanciaDerecha < self.distanciaDerecha_med*1.2: #80 < self.distanciaDerecha_med < 100# Condicion de salida de la curva (RETRASADA)
            if (self.indice_ultima_posicion is 10):
                self.indice_ultima_posicion = 0
            self.ultimas_posiciones[self.indice_ultima_posicion] = (self.left_points_up[0]+self.right_points_up[0])/2
            self.indice_ultima_posicion += 1

            self.distanciaDerecha_arr = np.zeros(10)
            self.indice_ultima_posicion_3 = 0
            self.array3_listo = False

            self.dentroCurvaDerecha = False
            self.columnasDeseadas = []

        if self.dentroCurvaDerecha:
            print(self.up_point)
            cv2.line(self.frameProcesado,(self.last,0),(self.last,320),(255,255,255),2)
            self.columnasDeseadas = [int(self.last/40)]
            
    def _dibujarGrilla(self):
        # Grid lines at these intervals (in pixels)
        # dx and dy can be different
        dx, dy = 40,40
        # Custom (rgb) grid color
        grid_color = [255,0,0]# [0,0,0]
        # Modify the image to include the grid
        self.frameProcesado[:,::dy,:] = grid_color
        self.frameProcesado[::dx,:,:] = grid_color

    def _calcularDistanciasLineaRecta(self):

        ubicacion_punto_central = (self.right_points_up[0] + self.left_points_up[0]) / 2        
        if self.dentroDeBocacalle:
            ubicacion_punto_central = (self.right_points_up_last + self.left_points_up_last) / 2
            distancia_al_centro = (self.width/2) - ubicacion_punto_central
        else:
            distancia_al_centro = (self.width/2) - ubicacion_punto_central

        #cv2.line(self.frameProcesado,(int(ubicacion_punto_central),0),(int(ubicacion_punto_central),240),(0,0,255),2)
        #cv2.line(self.frameProcesado,(int(320),0),(int(320),240),(0,255,255),2)
        
        if distancia_al_centro > 5:
            self.accionATomar = [1, 0, 0, 0]
        elif distancia_al_centro < -5:
            self.accionATomar = [0, 1, 0, 0]
        else:
            self.accionATomar = [0, 0, 1, 0]

        # DISTANCIA_A_MANTENER_IZQ = 116
        # DISTANCIA_A_MANTENER_DER = 58
        # distanciaIzquierda = (self.width/2) - self.left_points_up[0]
        # distanciaDerecha = self.right_points_up[0] - (self.width/2)
        # print("IZQUIERDA:   ", distanciaIzquierda, "    DERECHA:    ",distanciaDerecha)
        # if DISTANCIA_A_MANTENER_IZQ*1.05 < distanciaIzquierda:
        #     self.accionATomar = [1, 0, 1, 0]
        # elif DISTANCIA_A_MANTENER_IZQ > distanciaIzquierda:
        #     self.accionATomar = [0, 0, 0, 0]
        # if DISTANCIA_A_MANTENER_DER*1.05 < distanciaDerecha:
        #     self.accionATomar = [0, 1, 1, 0]
        # elif DISTANCIA_A_MANTENER_DER > distanciaDerecha:
        #     self.accionATomar = [0, 0, 0, 0]

    def _moverVehiculo(self):
        if self.accionATomar[0] == 1:
            print("Girar a la derecha")
        if self.accionATomar[1] == 1:
            print("Girar a la izquierda")
        if self.accionATomar[2] == 1:
            print("Ir hacia adelante")
        if self.accionATomar[3] == 1:
            print("Ir hacia atras")

    def _girarLineaPunteada(self):
        self.columnasDeseadas = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.filasDeseadas = [0,1,2,3,4,5]
        self.activarBuscarFramesLineaPunteada = True

    def run (self):
        while self.cap.isOpened():
            ret, frameCompleto = self.cap.read()
            if ret:
                frameOriginalRecortado = self._prepararFrame(frameCompleto)
                frameConFiltro = self._aplicarFiltrosMascaras(frameOriginalRecortado)

                cv2.imshow('frameParaProbar', frameConFiltro)
                
                self.frameProcesado = frameOriginalRecortado
                for fila in self.filasDeseadas: #Recorre de abajo para arriba, de izquierda a derecha
                    for columna in range(16):
                        porcionFrame, porcionFrameOriginal = self._obtenerPorcionesFrame(frameOriginalRecortado, frameConFiltro, fila, columna)
                        porcionFrameProcesado = self._procesarPorcionFrame(porcionFrame, porcionFrameOriginal, fila, columna)
                        self._reconstruirFrame(porcionFrameProcesado, fila, columna)
                for columna in self.columnasDeseadas: #Recorre de abajo para arriba, de izquierda a derecha
                    for fila in range(6):
                        if fila not in self.filasDeseadas:
                            porcionFrame, porcionFrameOriginal = self._obtenerPorcionesFrame(frameOriginalRecortado, frameConFiltro, fila, columna)
                            porcionFrameProcesado = self._procesarPorcionFrame(porcionFrame, porcionFrameOriginal, fila, columna)
                            self._reconstruirFrame(porcionFrameProcesado, fila, columna)
                
                if self.cartelDetectado:
                    if self.depositoDetectado is self.depositoABuscar:
                        self._girarLineaPunteada()
                    else:
                        self.filasDeseadas = [2, 5]
                        self._detectarBocacalle()
                        # Aca se limpia la bandera bocacalleDetectada mediante otra bandera dentroDeBocacalle. NO CONFUNDIR
                        if self.bocacalleDetectada:
                            self.dentroDeBocacalle = True
                        if (self.bocacalleDetectada == False and self.dentroDeBocacalle):
                            self.cartelDetectado = False
                            self.dentroDeBocacalle = False
                            self.filasDeseadas = [2]
                else: 
                    # Aca no se detecto ningun cartel y estoy pendiente a la espera de una curva a la derecha o izquierda
                    self._detectarCurvaDerecha()

                self._calcularDistanciasLineaRecta()
                # self._moverVehiculo()
                    
                self._dibujarGrilla()

                # Display the resulting frame
                cv2.imshow('frameResulting', self.frameProcesado)

                # Press Q on keyboard to  exit
                key = cv2.waitKey(10)
                if key == ord('q'):  # 25fps
                    break
                elif key == ord('k'): #Con esta tecla simulamos el cartel a detectar
                    self.cartelDetectado = True
            
            else: # Break the loop
                break

        # When everything done, release the video capture object
        self.cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()