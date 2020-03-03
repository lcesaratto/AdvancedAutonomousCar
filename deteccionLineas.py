import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
from pyzbar import pyzbar
from threading import Thread
import copy
# from controlPWM import *

'''
class VideoCamera(object):
    #320*240 original
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.cap.isOpened():
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.ret, self.frame = self.cap.read()
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.cap.release()
                return

    def read(self):
        # return the frame most recently read
        return self.ret, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
'''

class VehiculoAutonomo (object):
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

        # Banderas de prueba
        self.depositoDetectado = -1
        self.depositoABuscar = -1
        self.activarBuscarFramesLineaPunteada = False

        # Deteccion de objetos
        self.net = self._cargarModelo()
        self.classes = self._cargarClases()
        # self.height = 480
        # self.width = 640
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.font = cv2.FONT_HERSHEY_PLAIN

        # Contador de tiempo
        self.tiempoDeEsperaInicial = -1
        self.contadorFrames = 0

        #Deteccion de Color Rojo
        self.RojoDetectado = 0

        #Deteccion Linea Verde
        self.ubicacion_punto_verde = 0
    
    def _leer_qr(self, frame):
        barcodes = pyzbar.decode(frame)
        return barcodes

    def _detectarRojo(self,frame):
        #Defino parametros HSV para detectar color rojo 
        lower_red = np.array([170, 179, 0])
        upper_red = np.array([255, 255, 255])

        #Aplico filtro de color con los parametros ya definidos
        hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
        
        #Proceso
        if np.mean(mask_red) > 15:
            self.RojoDetectado = 1

    def _cargarModelo(self):
        return cv2.dnn.readNet("4class_yolov3-tiny_final.weights", "4class_yolov3-tiny.cfg")
    
    def _cargarClases(self):
        classes = []
        with open("4classes.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def _buscarObjetos (self, frame, mostrarResultado=False, retornarBoxes=False, retornarConfidence=False, calcularFPS=False):
        if calcularFPS:
            tiempo_inicial = time.time()

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # When we perform the detection, it happens that we have more boxes for the same object, so we should use another function to remove this “noise”.
        # It’s called Non maximum suppresion.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        cantidadEliminados = 0
        for i in range(len(boxes)):
                if not i in indexes:
                    del boxes[i-cantidadEliminados]
                    del confidences[i-cantidadEliminados]
                    del class_ids[i-cantidadEliminados]
                    cantidadEliminados += 1

        if calcularFPS:
            print(1/(time.time()-tiempo_inicial))

        if mostrarResultado:
            frameMostrado = copy.deepcopy(frame)
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                if label == 'SemaforoRojo' or label== 'SemaforoDos':
                    color = self.colors[0] #indice va de 0 a 3 para 4 clases
                elif label == 'SemaforoVerde':
                    color = self.colors[1] #indice va de 0 a 3 para 4 clases
                elif label == 'CartelUno':
                    color = self.colors[2] #indice va de 0 a 3 para 4 clases
                elif label == 'CartelCero':
                    color = self.colors[3] #indice va de 0 a 3 para 4 clases
                cv2.rectangle(frameMostrado, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frameMostrado, label, (x, y + 30), self.font, 3, color, 2)

            cv2.imshow('buscandoObjetos',frameMostrado)
            
        if not retornarBoxes and not retornarConfidence:
            return class_ids
        if retornarBoxes and retornarConfidence:
            return class_ids, boxes, confidences
        elif retornarBoxes:
            return class_ids, boxes
        else:
            return class_ids, confidences

    def _abrirCamara (self):
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        #cap = cv2.VideoCapture('Videos/20200107_163552.mp4')
        cap = cv2.VideoCapture(0)#('Videos/WhatsApp Video 2019-10-12 at 6.19.29 PM(2).mp4')
        # cap = VideoCamera()
        # cap.start()
        # time.sleep(2)
        # Check if camera opened successfully
        #if not cap.isOpened():
        #    print("Error opening video stream or file")
        return cap

    def _obtenerParametrosFrame(self):
        # fps =  30#
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # width = 640#
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        # height = 480#
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return fps, width, height

    def _prepararFrame (self, frame):
        # frame = cv2.flip(frame, flipCode=-1)
        frame = frame[int(self.height*0.5):int(self.height),0:int(self.width)]#frame = frame[0:int(self.height*0.5),0:int(self.width)]
        return frame

    def _aplicarFiltrosMascaras (self, frame):
        #Defino parametros HLS o HSV para eliminar fondo 
        lower_black = np.array([0, 0, 0]) #108 6 17     40 16 37
        upper_black = np.array([255, 255, 100 ]) #HSV 255, 255, 90 #HLS[150, 75, 255]

        #Aplico filtro de color con los parametros ya definidos
        hsv_right = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #BRG2HLS
        mask_right = cv2.inRange(hsv_right, lower_black, upper_black) #frame
        mask_right = np.invert(mask_right)
        res_right = cv2.bitwise_and(frame, frame, mask=mask_right)
        #cv2.imshow('framefilter', res_right)

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
        frame_e = cv2.erode(res_right, kernel, iterations=1)
        gray = cv2.cvtColor(frame_e, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.medianBlur(gray, 5)
        dst2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3)
        # cv2.imshow('framefilterdif', dst2)
        dif_gray_right = cv2.GaussianBlur(dst2, (11, 11), 0) #(mask, (5, 5), 0)
        #cv2.imshow('framefilterdif', frame_e)
        canny_right = cv2.Canny(dif_gray_right, 25, 175) # 25, 175)
        kernel = np.ones((7,7), np.uint8)
        canny_right = cv2.morphologyEx(canny_right, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((3,3), np.uint8)
        canny_right = cv2.erode(canny_right, kernel, iterations=1)
        cv2.imshow('framefilter', canny_right)
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

    def _detectarLineaVerde(self):
        #Corto el frame
        frame = self.frameProcesado
        frame = frame[100:320,0:int(self.width)]
        #Defino parametros HSV para detectar color verde 
        lower_green = np.array([20, 60, 100])
        upper_green = np.array([80, 230, 140])

        #Aplico filtro de color con los parametros ya definidos
        hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
        y, x = np.where(mask_green == 255)
        x_mid= statistics.median(x)
        x_mid_int=int(round(x_mid))
        self.ubicacion_punto_verde = x_mid_int
    
    def _calcularDistanciasLineaRecta(self):

        ubicacion_punto_central = (self.right_points_up[0] + self.left_points_up[0]) / 2        
        if self.dentroDeBocacalle:
            ubicacion_punto_central = (self.right_points_up_last + self.left_points_up_last) / 2
            distancia_al_centro = (self.width/2) - ubicacion_punto_central
        elif self.cartelDetectado:
            ubicacion_punto_central = self.ubicacion_punto_verde
            distancia_al_centro = (self.width/2) - ubicacion_punto_central

        else:
            distancia_al_centro = (self.width/2) - ubicacion_punto_central

        #cv2.line(self.frameProcesado,(int(ubicacion_punto_central),0),(int(ubicacion_punto_central),240),(0,0,255),2)
        #cv2.line(self.frameProcesado,(int(320),0),(int(320),240),(0,255,255),2)
        
        if distancia_al_centro > 5:
            giroDerecha()
        elif distancia_al_centro < -5:
            giroIzquierda()
        else:
            forward()

    def _girarLineaPunteada(self):
        self.columnasDeseadas = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.filasDeseadas = [0,1,2,3,4,5]
        self.activarBuscarFramesLineaPunteada = True

    def comenzar(self):
        while self.cap.isOpened():
            ret, frameCompleto = self.cap.read()
            if ret:
        # while True:
        #     ret, frameCompleto = self.cap.read()
        #     if ret:
                self.tiempoDeEsperaInicial = 0
                self.depositoABuscar = 0
                if self.depositoABuscar == -1:
                    cv2.imshow('buscandoQR', frameCompleto)
                    cv2.waitKey(10)
                    barcodes = self._leer_qr(frameCompleto)
                    if barcodes:
                        cv2.destroyWindow('buscandoQR')
                        barcodeData = barcodes[0].data.decode("utf-8")
                        print("Deposito a buscar: ", barcodeData)
                        self.depositoABuscar = 0
                        self.tiempoDeEsperaInicial = time.time()

                elif (time.time()-self.tiempoDeEsperaInicial) > 5 and self.tiempoDeEsperaInicial != -1:

                    tiempoInicialFPS = time.time()
                    self._detectarRojo(frameCompleto)
                    if self.RojoDetectado == 1:
                        if self.contadorFrames is 4:
                            class_ids = self._buscarObjetos(frameCompleto, mostrarResultado=False, retornarBoxes=False, retornarConfidence=False, calcularFPS=False)
                            print('Objetos detectados: ', class_ids)
                            if class_ids:
                                self.RojoDetectado = 0
                                if 2 in class_ids:
                                    self.cartelDetectado = True
                                    self.depositoDetectado = 0
                                elif 3 in class_ids:
                                    self.cartelDetectado = True
                                    self.depositoDetectado = 1
                            self.contadorFrames = 0
                        self.contadorFrames += 1

                    frameOriginalRecortado = self._prepararFrame(frameCompleto)
                    frameConFiltro = self._aplicarFiltrosMascaras(frameOriginalRecortado)

                    # cv2.imshow('frameParaProbar', frameConFiltro)
                    
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
                            self.filasDeseadas = []
                            self._detectarLineaVerde()
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
                    self.filasDeseadas = []
                    self.cartelDetectado = 1
                    self._detectarLineaVerde()
                    self._calcularDistanciasLineaRecta()
                    # self._moverVehiculo()
                        
                    self._dibujarGrilla()

                    # Display the resulting frame
                    cv2.imshow('frameResulting', self.frameProcesado)

                    # Press Q on keyboard to  exit
                    key = cv2.waitKey(10)
                    if key == ord('q'):  # 25fps
                        break
                    # elif key == ord('k'): #Con esta tecla simulamos el cartel a detectar
                    #     self.cartelDetectado = True

                    print("FPS: ", (1/(time.time()-tiempoInicialFPS)))
            else: # Break the loop
                break

        # When everything done, release the video capture object
        self.cap.stop()
        # Closes all the frames
        cv2.destroyAllWindows()
        exit()

if __name__ == "__main__":
    vehiculoAutonomo = VehiculoAutonomo()
    vehiculoAutonomo.comenzar()