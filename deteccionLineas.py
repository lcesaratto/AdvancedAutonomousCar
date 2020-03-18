import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
from pyzbar import pyzbar
import copy
from controlPWM import *

class VehiculoAutonomo (object):
    def __init__(self):
        self.cap = self._abrirCamara()
        self.fps, self.width, self.height = self._obtenerParametrosFrame()
        # Frame con el que trabajan todos los metodos
        self.frameProcesado = []
        # Especificamos mediante los siguientes array que porciones de frame procesar
        self.filasDeseadas = [1]
        self.columnasDeseadas = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        # Variables y banderas utilizadas cuando se detecta la bocacalle
        self.bocacalleDetectada=False
        self.right_points_up_arr = np.zeros(10)
        self.left_points_up_arr = np.zeros(10)
        self.right_points_up_med = 0
        self.left_points_up_med = 0
        self.left_points_up_last = np.array([0, 0])
        self.right_points_up_last = np.array([0, 0])
        self.dentroDeBocacalle = False
        # Array para almacenar las ultimas 10 posiciones del vehiculo
        self.ultimas_posiciones = np.zeros(10)
        self.indice_ultima_posicion_2 = 0
        # Cuando se detecta el cartel o se pulsa la letra K
        self.cartelDetectado = False
        # Ubicacion de los puntos a las lineas laterales
        self.right_points_up = np.array([0, 0]) # Para la fila 2
        self.left_points_up = np.array([0, 0])
        self.right_points_up_2 = np.array([0, 0]) # Para la fila 5
        self.left_points_up_2 = np.array([0, 0])
        # Banderas de prueba
        self.depositoHallado = -1
        self.depositoABuscar = -1
        # Deteccion de objetos
        self.net = self._cargarModelo()
        self.classes = self._cargarClases()
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.font = cv2.FONT_HERSHEY_PLAIN
        # Contador de tiempo
        self.tiempoDeEsperaInicial = -1
        #Deteccion de Color Rojo
        self.RojoDetectado = 0
        #Deteccion Linea Verde
        self.ubicacion_punto_verde = 0
        #PWM
        self.miPwm = iniciarPWM() # ToDo: Descomentar esta linea
        self.ultima_distancia = 0
        #Luminosidad Ambiente
        self.multiplicadorLuminosidadAmbiente = 2
    
    def _abrirCamara (self):
        # Create a VideoCapture object and read from input file
        return cv2.VideoCapture(0)
    
    def _obtenerParametrosFrame(self):
        # fps =  30
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # width = 640
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        # height = 480
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return fps, width, height

    def _cargarModelo(self):
        return cv2.dnn.readNet("4class_yolov3-tiny_final.weights", "4class_yolov3-tiny.cfg")
    
    def _cargarClases(self):
        classes = []
        with open("4classes.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def _obtenerLuminosidadAmbiente(self, frame, minLum=120, maxLum=160, minMul=2, maxMul=3):
        lower = np.array([100, 100, 100])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(frame, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        frame = cv2.cvtColor(res, cv2.COLOR_BGR2HLS)
        L = frame[:,:,1]
        L = L[L != 0]
        Lprom = np.average(L)

        m = -(maxMul - minMul)/(maxLum - minLum)
        h = minMul - m * maxLum

        return (Lprom * m + h)
    
    def _leer_qr(self, frame):
        cv2.imshow('buscandoQR', frame)
        cv2.waitKey(10)
        barcodes = pyzbar.decode(frame)
        if barcodes:
            cv2.destroyWindow('buscandoQR')
            barcodeData = barcodes[0].data.decode("utf-8")
            print("Deposito a buscar: ", barcodeData)
            self.depositoABuscar = 0
            self.tiempoDeEsperaInicial = time.time()

    def _detectarRojo(self,frame):
        #Defino parametros HSV para detectar color rojo 
        lower_red = np.array([170, 179, 0])
        upper_red = np.array([255, 255, 255])

        #Aplico filtro de color con los parametros ya definidos
        hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
        
        #Proceso
        if np.mean(mask_red) > 15:
            return True

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

    def _analizarFrameForFilasColumnas(self, frame):
        # Cortamos el frame a la mitad inferior
        # frameOriginalRecortado = self._prepararFrame(frame)
        # Aplicamos filtros, entre ellos canny
        frameConFiltro = self._aplicarFiltrosMascaras(frame)
        # cv2.imshow('frameParaProbar', frameConFiltro)

        # Aca solo creo un numpy array con las mismas dimensiones que el frame original recortado
        self.frameProcesado = frame

        # Analizo con hough las filas o columnas 
        for fila in self.filasDeseadas: #Recorre de abajo para arriba, de izquierda a derecha
            # for columna in range(16):
            for columna in self.columnasDeseadas:
                porcionFrameConFiltro, porcionFrame = self._obtenerPorcionesFrame(frame, frameConFiltro, fila, columna)
                porcionFrameProcesado = self._procesarPorcionFrame(porcionFrameConFiltro, porcionFrame, fila, columna) #ToDo: si comentamos la linea que reconstruye el retorno no es necesario
                # self._reconstruirFrame(porcionFrameProcesado, fila, columna)
        # for columna in self.columnasDeseadas: #Recorre de abajo para arriba, de izquierda a derecha
        #     for fila in range(6):
        #         if fila not in self.filasDeseadas:
        #             porcionFrame, porcionFrame = self._obtenerPorcionesFrame(frame, frameConFiltro, fila, columna)
        #             porcionFrameProcesado = self._procesarPorcionFrame(porcionFrame, porcionFrame, fila, columna)
        #             self._reconstruirFrame(porcionFrameProcesado, fila, columna)

    # def _prepararFrame (self, frame):
    #     # frame = cv2.flip(frame, flipCode=-1)
    #     frame = frame[int(self.height*0.5):int(self.height),0:int(self.width)]#frame = frame[0:int(self.height*0.5),0:int(self.width)]
    #     return frame

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
        #cv2.imshow('framefilter', canny_right)
        return canny_right

    def _obtenerPorcionesFrame (self, frame, frameConFiltro, row, column):
        frameConFiltroCut=frameConFiltro[(440-row*40):(480-row*40),(40*column):(40+40*column)]
        frameCut=frame[(440-row*40):(480-row*40),(40*column):(40+40*column)]
        return frameConFiltroCut, frameCut

    def _procesarPorcionFrame (self, porcionFrameConFiltro, porcionFrame, fila, columna):

        try:
            # porcionFrame = canny_right
            ## Aplico Transformada de Hough
            lines = cv2.HoughLinesP(porcionFrameConFiltro, 1, np.pi / 180, 10, minLineLength=0, maxLineGap=1) #(canny, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=150)

            # Draw lines on the image
            if lines is not None:
                cant_lineas=len(lines)
                # print(cant_lineas)
                
                if cant_lineas>100: # print('Se detecto una curva')
                    porcionFrame[:,:,2] = porcionFrame[:,:,2] + 30 #Pintamos de color rojo las porciones con curvas
                else: # print('Se detecto una linea')
                    x1M=0
                    x1M_2=0
                    y1M=0

                    for line in lines:
                        x1, y1, x2, y2 = line[0]

                        if(x1>x1M and fila==1):
                            x1M=x1
                            self.right_points_up = [x1+40*columna, y1+40*(5-fila)]
                            self.right_points_down = [x2, y2]
                        if(columna<7 and fila==1):
                            # print(x1+40*column)
                            self.left_points_up = [x1+40*columna, y1+40*(5-fila)]
                            self.left_points_down = [x2, y2]
                        if(x1>x1M_2 and fila==10):
                            x1M_2=x1
                            self.right_points_up_2 = [x1+40*columna, y1+40*(5-fila)]
                            self.right_points_down_2 = [x2, y2]
                        if(columna<7 and fila==10):
                            # print(x1+40*column)
                            self.left_points_up_2 = [x1+40*columna, y1+40*(5-fila)]
                            self.left_points_down_2 = [x2, y2] 

                        # Guardamos la posicion del punto superior (frente al vehiculo)
                        # if self.dentroCurvaDerecha:
                        #     if (columna is int(self.last/40)) and y1>y1M:
                        #     # if (self.last*0.7< (x1+40*columna) < self.last*1.3) and y1>y1M:
                        #         y1M=y1
                        #         self.up_point = [self.last,y1+40*(5-fila)]
                               
                        # x1m=x1
                        # right_points = [(x1,y1), (x2,y2)]

                        cv2.line(porcionFrame,(x1,y1),(x2,y2),(0,0,255),2) # ToDo: Comentar esta linea
                        # cv2.line(porcionFrame,(porcionFrame.shape[1]-1,x1),(0,y1),255,2)

                    # ToDO: comentar las dos lineas siguientes que pintan porciones de frame
                    porcionFrame[:,:,0] = porcionFrame[:,:,0]+30 #Si se detecto una linea pintar el frame de color azul
            else:
                porcionFrame[:,:,1] = porcionFrame[:,:,1] + 30 #Si no se detecto ninguna linea, pintamos de color verde  

        except Exception as e:
            print(e)
        return porcionFrame # ToDo: Este retorno no es mas necesario en la version final

    def _reconstruirFrame(self, porcionFrameProcesado, fila, columna):
        for x in range(40): #filas
            for j in range(40): #columnas
                self.frameProcesado[200-fila*40+x][0+40*columna+j] = porcionFrameProcesado[x][j]

    def _detectarBocacalle(self):
        # if self.bocacalleDetectada:
        #     cv2.line(self.frameProcesado,(self.left_points_up_last,140),(self.right_points_up_last,140),(0,255,0),2)
        # else:
        #     cv2.line(self.frameProcesado,(self.left_points_up[0],140),(self.right_points_up[0],140),(255,255,255),2)

        # cv2.line(self.frameProcesado,(self.left_points_up_2[0],20),(self.right_points_up_2[0],20),(255,255,255),2)

        dist_line_down = self.right_points_up[0] - self.left_points_up[0]
        dist_line_up = self.right_points_up_2[0] - self.left_points_up_2[0]

        print('dist_line_down: ', dist_line_down)
            
        if ((dist_line_down > 250)): #Si en la fila inferior de la camara, la distancia entre las lineas negras laterales es mayor a 200
            if not self.bocacalleDetectada: #Y la bandera no esta seteada aun
                self.right_points_up_last = self.right_points_up_med #Guarda el ultimo valor de la mediana cuando entra en bocacalle por unica vez
                self.left_points_up_last = self.left_points_up_med 
            self.bocacalleDetectada=True #Seteamos la bandera
            # ToDo: Aca podemos setear la fila superior como la deseada
            self.filasDeseadas = [1,10]
            if dist_line_up < 250: #Si la distancia entre lineas negras en la fila superior de la camara, es menor a 200, ya encontro la proxima calle
                self.right_points_up_last = self.right_points_up_2[0] #Voy guardando los puntos superiores
                self.left_points_up_last = self.left_points_up_2[0]

        else: #Si no se detecto bocacalle todavia
            self.filasDeseadas = [1]
            if (self.indice_ultima_posicion_2 is 10): #Resetea el indice del buffer circular
                self.indice_ultima_posicion_2 = 0
            self.right_points_up_arr[self.indice_ultima_posicion_2] = self.right_points_up[0] #Agrega los valores al buffer circular
            self.left_points_up_arr[self.indice_ultima_posicion_2] = self.left_points_up[0]
            self.right_points_up_med = int(statistics.median(self.right_points_up_arr)) #Calculamos la mediana del buffer con los 10 ultimos puntos
            self.left_points_up_med = int(statistics.median(self.left_points_up_arr))
            self.indice_ultima_posicion_2 += 1

            if ((self.right_points_up_med*0.9 < self.right_points_up[0] < self.right_points_up_med*1.1) and (self.left_points_up_med*0.9 < self.left_points_up[0] < self.left_points_up_med*1.1)):
                self.bocacalleDetectada=False
            
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
        frame = frame[100:320,0:int(self.width)] #100 320
        #Defino parametros HSV para detectar color verde 
        lower_green_noche = np.array([20, 60, 100])
        upper_green_noche = np.array([80, 230, 140])
        lower_green_dia = np.array([20, 40, 100])
        upper_green_dia = np.array([80, 230, 140])
        lower_green = np.array([40, int(20*self.multiplicadorLuminosidadAmbiente), 100]) #lower_green = np.array([40, int(20*self.multiplicadorLuminosidadAmbiente), 100])
        upper_green = np.array([80, 230, 140])
        
        #Aplico filtro de color con los parametros ya definidos
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
        #kernel = np.ones((3,3), np.uint8)
        #mask_green_a = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        #kernel = np.ones((7,7), np.uint8)
        #mask_green_e = cv2.dilate(mask_green, kernel, iterations=1)
        #kernel = np.ones((11,11), np.uint8)
        #mask_green_c = cv2.morphologyEx(mask_green_e, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('frameResulting', mask_green)
        y, x = np.where(mask_green == 255)
        try:
            if len(x) < 100:
                x_mid = 0
            else:
                x_mid= statistics.median(x)
        except:
            x_mid = 0
        x_mid_int=int(round(x_mid))
        self.ubicacion_punto_verde = x_mid_int
    
    def _tomarDecisionMovimiento(self):
        # Si detecto la bocacalle me preparo para doblar o seguir, esta bandera se limpia sola cuando terminamos de cruzar
        if self.bocacalleDetectada:
            if (self.depositoHallado == self.depositoABuscar) and (self.depositoHallado != -1):
                self._moverVehiculoEnLineaVerde() #Doblarp
            elif (self.depositoHallado != self.depositoABuscar) and (self.depositoHallado != -1):
                self._moverVehiculoCruzarBocacalle() #Seguir derecho

        # Si no detecto bocacalle estoy girando en la linea verde nuevamente o recien comenzando el programa
        else:
            self._moverVehiculoEnLineaVerde() #Sigue derecho
        
    def _moverVehiculoEnLineaVerde(self):

        ubicacion_punto_central = (self.right_points_up[0] + self.left_points_up[0]) / 2        
        if self.dentroDeBocacalle:
            ubicacion_punto_central = (self.right_points_up_last + self.left_points_up_last) / 2
            distancia_al_centro = (self.width/2) - ubicacion_punto_central
        elif self.cartelDetectado: # ToDo: Ver esta condicion, probablemente ya no sirva
            ubicacion_punto_central = self.ubicacion_punto_verde
            distancia_al_centro = (self.width/2) - ubicacion_punto_central

        else:
            distancia_al_centro = (self.width/2) - ubicacion_punto_central
        
        #cv2.line(self.frameProcesado,(int(ubicacion_punto_central),0),(int(ubicacion_punto_central),240),(0,0,255),2)
        #cv2.line(self.frameProcesado,(int(320),0),(int(320),240),(0,255,255),2)
        print(distancia_al_centro)
        vel_brusca_max=3000
        vel_brusca_min=2200
        vel_suave_max=2200
        vel_suave_min=1700
        if abs(distancia_al_centro) == 320:
            if self.contandoFramesParado != 10:
                stop(self.miPwm)
                self.contandoFramesParado += 1
                self.contandoFramesBackward = 0
            #elif self.contandoFramesBackward != 3:
            else:
                #backward(self.miPwm, 1700)
                if self.ultima_distancia >= 0:
                    giroDerechaBrusco(self.miPwm, vel_brusca_min, vel_brusca_min)
                else:
                    giroIzquierdaBrusco(self.miPwm, vel_brusca_min, vel_brusca_min)
                self.contandoFramesBackward += 1
                if self.contandoFramesBackward == 3:
                    self.contandoFramesParado = 0
        else:
            self.contandoFramesParado = 0
            if distancia_al_centro > 50 and abs(distancia_al_centro) < 200:
                print("Izquierda Suave")
                giroIzquierdaSuave(self.miPwm, vel_suave_max, vel_suave_min)
            elif distancia_al_centro < -50 and abs(distancia_al_centro) < 200:
                giroDerechaSuave(self.miPwm, vel_suave_min, vel_suave_max)
                print("Derecha Suave")
            elif 320 > abs(distancia_al_centro) >= 200 and self.ultima_distancia <= 0:
                    giroDerechaBrusco(self.miPwm, vel_brusca_min, vel_brusca_max)
                    print("Derecha Brusco")
            elif 320 > abs(distancia_al_centro) >= 200 and self.ultima_distancia > 0:
                    giroIzquierdaBrusco(self.miPwm, vel_brusca_max,vel_brusca_min)
                    print("Izquierda Brusco")
            else:
                forward(self.miPwm, vel_suave_min)
        if abs(distancia_al_centro) < 200:
            self.ultima_distancia = distancia_al_centro

    def _moverVehiculoCruzarBocacalle(self):
        vel_suave_min=1700
        forward(self.miPwm, vel_suave_min)

    def comenzar(self):
        # En el proximo loop calcularemos la intensidad de luz ambiente para ajustar filtros
        contadorInicial = 0
        sumatoriaMultiplicador = 0
        while self.cap.isOpened():
            ret, frameCompleto = self.cap.read()
            if ret:
                if 1 < contadorInicial < 5:
                    sumatoriaMultiplicador += self._obtenerLuminosidadAmbiente(frameCompleto)
                    contadorInicial += 1
                elif contadorInicial == 5:
                    self.multiplicadorLuminosidadAmbiente = sumatoriaMultiplicador/3
                    break
                else:
                    contadorInicial += 1

        # Aca comienza el programa automaticamente
        while self.cap.isOpened():
            ret, frameCompleto = self.cap.read()
            if ret:
                self.tiempoDeEsperaInicial = 0 # ToDo: Borrar esta linea
                self.depositoABuscar = 0 # ToDo: Borrar esta linea

                # Aca corremos la funcion que busca un codigo qr en la imagen para comenzar
                if self.depositoABuscar == -1:
                    self._leer_qr(frameCompleto)

                # Si se cumple el tiempo de espera inicial luego de leer el QR, comenzamos a movernos
                elif (time.time()-self.tiempoDeEsperaInicial) > 5 and self.tiempoDeEsperaInicial != -1:
                    tiempoInicialFPS = time.time()

                    # Comenzamos buscando objetos si se detecta la senda peatonal roja
                    if False:
                        if not self.cartelDetectado:
                            if self._detectarRojo(frameCompleto): #ToDO: Falta hacer que solo busque cuando ve la senda por primera vez
                                class_ids = self._buscarObjetos(frameCompleto)
                                print('Objetos detectados: ', class_ids) # ToDo: Borrar print
                                if class_ids:
                                    if 2 in class_ids:
                                        tiempoInicialLuegoDeDeteccionCartel = time.time()
                                        self.cartelDetectado = True
                                        self.depositoHallado = 0
                                    elif 3 in class_ids:
                                        tiempoInicialLuegoDeDeteccionCartel = time.time()
                                        self.cartelDetectado = True
                                        self.depositoHallado = 1
                        else:
                            if not self._detectarRojo(frameCompleto):
                                # Espero 10segundos para borrar la bandera de cartel detectado
                                if (time.time()-tiempoInicialLuegoDeDeteccionCartel) > 10: 
                                    self.cartelDetectado = False
                    else:
                        self.cartelDetectado = True
                        self.depositoHallado = 0
                    
                    # Aca se aplican todos los filtros al frame, luego hough y se guardan las posiciones de las lineas 
                    # encontradas para luego utilizar en self._detectarBocacalle()
                    self._analizarFrameForFilasColumnas(frameCompleto)
                    
                    # Busco constantemente el punto central de la linea verde
                    self._detectarLineaVerde()

                    # Busco constantemente la bocacalle y su fin
                    self._detectarBocacalle()
                    # self.bocacalleDetectada = False

                    # En base a los resultados de self._detectarBocacalle() decido si seguir la linea verde o cruzar la bocacalle
                    self._tomarDecisionMovimiento()

                    # Mostrar grilla
                    self._dibujarGrilla()
                    # Display the resulting frame
                    #cv2.imshow('frameResulting', self.frameProcesado)

                    # Press Q on keyboard to  exit
                    key = cv2.waitKey(10)
                    if key == ord('q') or key == ord('Q'):
                        stop(self.miPwm)
                        break

                    print("FPS: ", (1/(time.time()-tiempoInicialFPS)))
            else: # Break the loop
                break

        # When everything done, release the video capture object
        self.cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        exit()

if __name__ == "__main__":
    vehiculoAutonomo = VehiculoAutonomo()
    vehiculoAutonomo.comenzar()
