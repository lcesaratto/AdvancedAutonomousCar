import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
from pyzbar import pyzbar
import copy
from controlPWM import procesoAuxiliar
from multiprocessing import Process, Pipe
from threading import Thread


def procesoPrincipal(enviar1):

    class VehiculoAutonomo (object):
        def __init__(self):
            self.frameCompleto = []
            self.girandoHaciaDerecha = False

            self.cap = self._abrirCamara()
            self.fps, self.width, self.height = self._obtenerParametrosFrame()
            # Frame con el que trabajan todos los metodos
            self.frameProcesado = []
            self.mask_green = []
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
            self.contandoFramesParado = 0
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
            self.listoParaReiniciar = False
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
            self.stoppingCounter = 0
            self.stoppingCounterMax=2
            #PWM
            # self.miPwm = iniciarPWM() # ToDo: Descomentar esta linea
            self.ultima_distancia = 0
            #Luminosidad Ambiente
            self.multiplicadorLuminosidadAmbiente = 2
            self.indiceCircular = 0
            self.arrayCircular = np.zeros(5)
        
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
                # print('asdad$$$$$$$$$$$$$$$$$$$$$$$           ', barcodes[0])
                cv2.destroyWindow('buscandoQR')
                barcodeData = barcodes[0].data.decode("utf-8")
                if barcodeData[0] == 'I':
                    print("Deposito a buscar: ", barcodeData[1])
                    self.depositoABuscar = barcodeData[1]
                    self.tiempoDeEsperaInicial = time.time()

        def _buscar_qr(self, frame):
            # global frameGlobal
            # frameGlobal=frame
            # enviar2.send('frame') #Todo: make frame global to increase speed
            barcodes = pyzbar.decode(frame)
            # barcodes = ['A1']
            if barcodes:
                qr_encontrado = barcodes[0].data.decode("utf-8")
                # print(qr_encontrado)
                # qr_encontrado = 'A1'
                # ToDo: a qr_encontrado falta sacarle la 'F' de fin para compara contra self.depositoABuscar
                if qr_encontrado[0] == 'F' and qr_encontrado[1] == self.depositoABuscar and not self.listoParaReiniciar:
                # if qr_encontrado == self.depositoABuscar:
                    print('Dejando paquete!!')
                    self.depositoHallado = -1
                    enviar1.send('stopAndIgnore')
                    # controladorPwm.actualizarOrden('stop')
                    # stop(self.miPwm)
                    # cv2.waitKey(15000)
                    self.listoParaReiniciar = True
                if qr_encontrado[0] == 'P' and self.listoParaReiniciar == True:
                    print("inicio hallado")
                    enviar1.send('stop')
                    # controladorPwm.actualizarOrden('stop')
                    # stop(self.miPwm)
                    self.listoParaReiniciar = False
                    self.tiempoDeEsperaInicial = -1
                    self.depositoABuscar = -1

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

        def _analizarFrameForFilasColumnas(self, frameOriginal):
            # Cortamos el frame a la mitad inferior
            # frameOriginalRecortado = self._prepararFrame(frame)
            # Aplicamos filtros, entre ellos canny
            frame = copy.deepcopy(frameOriginal)

            frameConFiltro = self._aplicarFiltrosMascaras(frame)
            # cv2.imshow('frameParaProbar', frameConFiltro)

            # Aca solo creo un numpy array con las mismas dimensiones que el frame original recortado
            self.frameProcesado = frame

            # Analizo con hough las filas o columnas 
            for fila in self.filasDeseadas: #Recorre de abajo para arriba, de izquierda a derecha
                # for columna in range(16):
                for columna in self.columnasDeseadas:
                    porcionFrameConFiltro, porcionFrame = self._obtenerPorcionesFrame(frame, frameConFiltro, fila, columna)
                    self._procesarPorcionFrame(porcionFrameConFiltro, porcionFrame, fila, columna) #ToDo: si comentamos la linea que reconstruye el retorno no es necesario
                    # self._reconstruirFrame(porcionFrameProcesado, fila, columna)
            # for columna in self.columnasDeseadas: #Recorre de abajo para arriba, de izquierda a derecha
            #     for fila in range(6):
            #         if fila not in self.filasDeseadas:
            #             porcionFrame, porcionFrame = self._obtenerPorcionesFrame(frame, frameConFiltro, fila, columna)
            #             porcionFrameProcesado = self._procesarPorcionFrame(porcionFrame, porcionFrame, fila, columna)
            #             self._reconstruirFrame(porcionFrameProcesado, fila, columna)

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

                            # cv2.line(porcionFrame,(x1,y1),(x2,y2),(0,0,255),2) # ToDo: Comentar esta linea
                            # cv2.line(porcionFrame,(porcionFrame.shape[1]-1,x1),(0,y1),255,2)

                        # ToDO: comentar las dos lineas siguientes que pintan porciones de frame
                        porcionFrame[:,:,0] = porcionFrame[:,:,0]+30 #Si se detecto una linea pintar el frame de color azul
                else:
                    porcionFrame[:,:,1] = porcionFrame[:,:,1] + 30 #Si no se detecto ninguna linea, pintamos de color verde  

            except Exception as e:
                print(e)
            # return porcionFrame # ToDo: Este retorno no es mas necesario en la version final

        def _reconstruirFrame(self, porcionFrameProcesado, fila, columna):
            for x in range(40): #filas
                for j in range(40): #columnas
                    self.frameProcesado[440-fila*40+x][0+40*columna+j] = porcionFrameProcesado[x][j]

        def _detectarBocacalle(self):
            if self.bocacalleDetectada:
                cv2.line(self.frameProcesado,(self.left_points_up_last,420),(self.right_points_up_last,420),(0,255,0),2)
            else:
                cv2.line(self.frameProcesado,(self.left_points_up[0],420),(self.right_points_up[0],420),(255,255,255),2)

            # cv2.line(self.frameProcesado,(self.left_points_up_2[0],20),(self.right_points_up_2[0],20),(255,255,255),2)

            dist_line_down = self.right_points_up[0] - self.left_points_up[0]
            dist_line_up = self.right_points_up_2[0] - self.left_points_up_2[0]

            # print('dist_line_down: ', dist_line_down)
                
            if ((dist_line_down > 350)): #Si en la fila inferior de la camara, la distancia entre las lineas negras laterales es mayor a 200
                if not self.bocacalleDetectada: #Y la bandera no esta seteada aun
                    self.right_points_up_last = self.right_points_up_med #Guarda el ultimo valor de la mediana cuando entra en bocacalle por unica vez
                    self.left_points_up_last = self.left_points_up_med 
                # print('BOCACALLE DETECTADA!')
                self.bocacalleDetectada=True #Seteamos la bandera
                # ToDo: Aca podemos setear la fila superior como la deseada
                self.filasDeseadas = [1,10]
                if dist_line_up < 270: #Si la distancia entre lineas negras en la fila superior de la camara, es menor a 200, ya encontro la proxima calle
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
                    # print('CALLE DETECTADA!')
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

        def _detectarLineaVerde(self, frameOriginal):
            #Corto el frame
            # frame = self.frameProcesado
            frame = frameOriginal[320:480,0:int(self.width)] #
            # cv2.imshow('FrameOriginalRecortado', frame)
            #Defino parametros HSV para detectar color verde 
            lower_green_noche = np.array([20, 60, 100])
            upper_green_noche = np.array([80, 230, 140])
            lower_green_dia = np.array([20, 40, 100])
            upper_green_dia = np.array([80, 230, 140])
            lower_green = np.array([40, int(20*self.multiplicadorLuminosidadAmbiente), 100]) #lower_green = np.array([40, int(20*self.multiplicadorLuminosidadAmbiente), 100])
            # lower_green = np.array([40, 50, 80])
            upper_green = np.array([80, 230, 140])
            
            #Aplico filtro de color con los parametros ya definidos
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
            self.mask_green = copy.deepcopy(mask_green)
            #kernel = np.ones((3,3), np.uint8)
            #mask_green_a = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
            #kernel = np.ones((7,7), np.uint8)
            #mask_green_e = cv2.dilate(mask_green, kernel, iterations=1)
            #kernel = np.ones((11,11), np.uint8)
            #mask_green_c = cv2.morphologyEx(mask_green_e, cv2.MORPH_CLOSE, kernel)
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
            # print(self.ubicacion_punto_verde)
            distancia_al_centro = (self.width/2) - self.ubicacion_punto_verde
            if x.size > 50 and distancia_al_centro < 320:    
                m,b = np.polyfit(x,y,1)
                print('pendiente:',m)
                if m < 0:
                    self.girandoHaciaDerecha = True
                else:
                    self.girandoHaciaDerecha = False
            # #     print('##########################' ,m,b)
            #     cv2.line(mask_green,(int(-b/m),0),(int((160-b)/m),160),(255,255,255), 3)
            # #     real_m = 160/(((160-b)/m)+b/m)
            # #     print(real_m)

            cv2.line(mask_green,(self.ubicacion_punto_verde,0),(self.ubicacion_punto_verde,480),(255,255,255), 2)
            # cv2.imshow('FiltroVerde', mask_green)
        
        def _tomarDecisionMovimiento(self):
            # Si detecto la bocacalle me preparo para doblar o seguir, esta bandera se limpia sola cuando terminamos de cruzar
            if self.bocacalleDetectada:
                if self.listoParaReiniciar:
                    self._moverVehiculoCruzarBocacalle()
                elif (self.depositoHallado == self.depositoABuscar) and (self.depositoHallado != -1):
                    self._moverVehiculoEnLineaVerde() #Doblar
                elif (self.depositoHallado != self.depositoABuscar) and (self.depositoHallado != -1):
                    print('########################################################################CRUZAR')
                    self._moverVehiculoCruzarBocacalle() #Seguir derecho

            # Si no detecto bocacalle estoy girando en la linea verde nuevamente o recien comenzando el programa
            else:
                self._moverVehiculoEnLineaVerde() #Sigue derecho
            
        def _moverVehiculoEnLineaVerde(self):

            distancia_al_centro = (self.width/2) - self.ubicacion_punto_verde

            # self.stoppingCounter +=1
            # if self.stoppingCounter == 1:
            #     stop(self.miPwm)
            # else:
            #     if self.stoppingCounter >= self.stoppingCounterMax:
            #         self.stoppingCounter = 0
            
                #cv2.line(self.frameProcesado,(int(ubicacion_punto_central),0),(int(ubicacion_punto_central),240),(0,0,255),2)
                #cv2.line(self.frameProcesado,(int(320),0),(int(320),240),(0,255,255),2)
                # print(distancia_al_centro)

                # vel_brusca_max=2450
                # vel_brusca_min=1000
                # # vel_brusca_max_perdido=2500
                # # vel_brusca_min_perdido=1400
                # vel_suave_max=1400
                # vel_suave_min=700
                # vel_forward = 1200

            if abs(distancia_al_centro) == 320:
                if self.contandoFramesParado != 3:
                    # stop(self.miPwm)
                    enviar1.send('stop')
                    # controladorPwm.actualizarOrden('stop')
                    self.contandoFramesParado += 1
                    self.contandoFramesBackward = 0
                else:
                    # if self.girandoHaciaDerecha:
                    #     enviar1.send('giroBruDer')
                    # else:
                    #     enviar1.send('giroBruIzq')
                    # controladorPwm.actualizarOrden('backward')
                    # enviar1.send('backward')
                    # backward(self.miPwm, vel_forward)
                    print('/////////////////////////////////////////////////////////     ', self.ultima_distancia)

                    if self.ultima_distancia <= 0:
                    #     print('PERDIDO- Derecha brusco')
                        enviar1.send('giroBruDer')
                    #     giroDerechaBrusco(self.miPwm, vel_brusca_min_perdido, vel_brusca_min_perdido)
                    else:
                        enviar1.send('giroBruIzq')
                    #     print('PERDIDO- Izquierda brusco')
                    #     giroIzquierdaBrusco(self.miPwm, vel_brusca_min_perdido, vel_brusca_min_perdido)
                    self.contandoFramesBackward += 1
                    if self.contandoFramesBackward == 2:
                        self.contandoFramesParado = 0
            
            else:
                limite1 = 40
                limite2 = 170
                distancia_al_centro -= 11
                self.contandoFramesParado = 0
                # print('distancia al centro: ', distancia_al_centro, self.ubicacion_punto_verde)
                if distancia_al_centro > limite1 and abs(distancia_al_centro) < limite2:
                    # print("Izquierda Suave")
                    enviar1.send('giroSuaIzq')
                    # controladorPwm.actualizarOrden('giroSuaIzq')
                    # self.stoppingCounterMax = 2
                    # giroIzquierdaSuave(self.miPwm, vel_suave_max, vel_suave_min)
                elif distancia_al_centro < -limite1 and abs(distancia_al_centro) < limite2:
                    # giroDerechaSuave(self.miPwm, vel_suave_min, vel_suave_max)
                    enviar1.send('giroSuaDer')
                    # print("Derecha Suave")
                    # controladorPwm.actualizarOrden('giroSuaDer')
                    # self.stoppingCounterMax = 2
                elif 320 > abs(distancia_al_centro) >= limite2 and self.ultima_distancia <= 0:
                        # giroDerechaBrusco(self.miPwm, vel_brusca_min, vel_brusca_max)
                        # print("Derecha Brusco")
                        enviar1.send('giroBruDer')
                        # controladorPwm.actualizarOrden('giroBruDer')
                        # self.stoppingCounterMax = 2
                elif 320 > abs(distancia_al_centro) >= limite2 and self.ultima_distancia > 0:
                        # giroIzquierdaBrusco(self.miPwm, vel_brusca_max,vel_brusca_min)
                        # print("Izquierda Brusco")
                        enviar1.send('giroBruIzq')
                        # controladorPwm.actualizarOrden('giroBruIzq')
                        # self.stoppingCounterMax = 2
                else:
                    # self.stoppingCounterMax = 2
                    # forward(self.miPwm, vel_forward)
                    # print('forward')
                    enviar1.send('forward')
                    # controladorPwm.actualizarOrden('forward')

            if abs(distancia_al_centro) < 320:
                self.ultima_distancia = distancia_al_centro
                # print(self.ultima_distancia)

        def _moverVehiculoCruzarBocacalle(self):
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            # vel_suave_min=1700
            # forward(self.miPwm, vel_suave_min)
            enviar1.send('forwardLong')
            # controladorPwm.actualizarOrden('stop')
            # stop(self.miPwm)

        def _buscarDeposito(self):
            # frameMitad = self.frameCompleto[int(self.height*0.5):int(self.height),0:int(self.width)] #Mitad inferior
            self._buscar_qr(self.frameCompleto)

        def _actualizarValorSaturacion(self):
            self.arrayCircular[self.indiceCircular] = self._obtenerLuminosidadAmbiente(self.frameCompleto)
            self.indiceCircular += 1
            if self.indiceCircular == 5:
                self.indiceCircular = 0
            self.multiplicadorLuminosidadAmbiente = np.mean(self.arrayCircular)
            # print(self.multiplicadorLuminosidadAmbiente)
            # 

        def _detectarBocacalleVerde(self):

            frame = copy.deepcopy(self.frameCompleto)
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

            if len(x_up) > 200 and len(x_down) > 1000:
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
                diagonalNoCruza = False

            if suficientesPuntos and diagonalNoCruza:           
                self.bocacalleDetectada = True
                # enviar1.send('stopAndIgnore')
                # print('AAAAAAAAAAAAAAA BOCACALLE DETECTADA!')
            else:
                self.bocacalleDetectada = False

        def comenzar(self):
            # try:
            # En el proximo loop calcularemos la intensidad de luz ambiente para ajustar filtros
            while self.cap.isOpened():
                ret, frameCompleto = self.cap.read()
                if ret:
                    if 1 < self.indiceCircular < 7:
                        self.arrayCircular[self.indiceCircular-2] = self._obtenerLuminosidadAmbiente(frameCompleto)
                        self.indiceCircular += 1
                    elif self.indiceCircular == 7:
                        self.multiplicadorLuminosidadAmbiente = np.mean(self.arrayCircular)
                        self.indiceCircular = 0
                        break
                    else:
                        self.indiceCircular += 1

            # Aca comienza el programa automaticamente
            while self.cap.isOpened():
                ret, frameCompleto = self.cap.read()
                if ret:
                    self.frameCompleto = frameCompleto
                    # print('hola')
                    out.write(frameCompleto)
                    # self.tiempoDeEsperaInicial = 0 # ToDo: Borrar esta linea
                    # self.depositoABuscar = 0 # ToDo: Borrar esta linea
                    # self.listoParaReiniciar = False

                    # Aca corremos la funcion que busca un codigo qr en la imagen para comenzar
                    if self.listoParaReiniciar:
                        self._buscar_qr(frameCompleto)
                    else:
                        if self.depositoABuscar == -1:
                            self._leer_qr(frameCompleto)
                    
                    # Si se cumple el tiempo de espera inicial luego de leer el QR, comenzamos a movernos
                    if (time.time()-self.tiempoDeEsperaInicial) > 5 and self.tiempoDeEsperaInicial != -1:
                        tiempoInicialFPS = time.time()

                        # Comenzamos buscando objetos si se detecta la senda peatonal roja
                        if False:
                            if not self.cartelDetectado:
                                self._detectarRojo(frameCompleto)
                                if self.RojoDetectado: #ToDO: Falta hacer que solo busque cuando ve la senda por primera vez
                                    #cuando deja de ver rojo deja de buscar carteles, hay que hacer una bandera 
                                    # para dejarla levantada y mientras este levantada va a buscar carteles. una vez que encuentra un cartel se limpia
                                    class_ids = self._buscarObjetos(frameCompleto)
                                    print('Objetos detectados: ', class_ids) # ToDo: Borrar print
                                    if class_ids:
                                        if 2 in class_ids:
                                            tiempoInicialLuegoDeDeteccionCartel = time.time()
                                            self.RojoDetectado = False
                                            self.cartelDetectado = True
                                            self.depositoHallado = 0
                                        elif 3 in class_ids:
                                            tiempoInicialLuegoDeDeteccionCartel = time.time()
                                            self.RojoDetectado = False
                                            self.cartelDetectado = True
                                            self.depositoHallado = 1
                            else:
                                self._detectarRojo(frameCompleto)
                                if not self.RojoDetectado:
                                    # Espero 10segundos para borrar la bandera de cartel detectado
                                    if (time.time()-tiempoInicialLuegoDeDeteccionCartel) > 10:
                                        self.cartelDetectado = False
                        else:
                            self.cartelDetectado = True
                            self.depositoHallado = '1'
                        
                        # Aca se aplican todos los filtros al frame, luego hough y se guardan las posiciones de las lineas 
                        # encontradas para luego utilizar en self._detectarBocacalle()

                        Thread(target=self._actualizarValorSaturacion, args=()).start()

                        # print('////////////////////////////////////////////////////: ',self.multiplicadorLuminosidadAmbiente)
                        
                        # self._actualizarValorSaturacion(frameCompleto)

                        # print("FPS 1: ", (1/(time.time()-tiempoInicialFPS)))

                        # self._analizarFrameForFilasColumnas(frameCompleto)

                        # print("FPS 2: ", (1/(time.time()-tiempoInicialFPS)))

                        # Busco constantemente el punto central de la linea verde
                        self._detectarLineaVerde(frameCompleto)

                        # print("FPS 3: ", (1/(time.time()-tiempoInicialFPS)))

                        # Busco constantemente la bocacalle y su fin
                        # self._detectarBocacalle()
                        Thread(target=self._detectarBocacalleVerde, args=()).start()
                        # self.bocacalleDetectada= False
                        # self._detectarBocacalleVerde()

                        # print("FPS 4: ", (1/(time.time()-tiempoInicialFPS)))

                        # En base a los resultados de self._detectarBocacalle() decido si seguir la linea verde o cruzar la bocacalle
                        self._tomarDecisionMovimiento()

                        # print("FPS 5: ", (1/(time.time()-tiempoInicialFPS)))

                        Thread(target=self._buscarDeposito, args=()).start()

                        # self._buscarDeposito(frameCompleto)

                        # print("FPS 6: ", (1/(time.time()-tiempoInicialFPS)))

                        # Mostrar grilla
                        # self._dibujarGrilla()
                        # Display the resulting frame
                        # cv2.imshow('frameCompleto', frameCompleto)
                        cv2.imshow('filtroVerde', self.mask_green)

                        # Press Q on keyboard to  exit
                        key = cv2.waitKey(10)
                        if key == ord('q') or key == ord('Q'):
                            enviar1.send('stop')
                            # controladorPwm.actualizarOrden('stop')
                            cv2.waitKey(10)
                            enviar1.send('exit')
                            break

                        # print("FPS: ", (1/(time.time()-tiempoInicialFPS)))
                else: # Break the loop
                    break

            # When everything done, release the video capture object
            self.cap.release()
            out.release()
            # Closes all the frames
            cv2.destroyAllWindows()
            exit()
        
            # except Exception as e:
            #     print(e)
            #     self.cap.release()
            #     cv2.destroyAllWindows()
            #     stop(self.miPwm)
            #     exit()

    vehiculoAutonomo = VehiculoAutonomo()
    vehiculoAutonomo.comenzar()

def procesoAuxiliar2(recibir2):
    def loop():
        while True:
            # time.sleep(0.1)
            frame = recibir2.recv()
            # print('$$$$$$$$$$$$$$$$$$$$$$$$: ',frameGlobal)
            # barcodes = pyzbar.decode(frame)
            # print(barcodes.data)

    loop()

if __name__ == "__main__":
    out = cv2.VideoWriter('outputAut3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

    enviar1, recibir1 = Pipe()
    # enviar2, recibir2 = Pipe()
    
    P_principal = Process(target=procesoPrincipal, args=(enviar1,))
    P_auxiliar = Process(target=procesoAuxiliar, args=(recibir1,))
    # P_auxiliar2 = Process(target=procesoAuxiliar2, args=(recibir2,))
    
    P_principal.start()
    P_auxiliar.start()
    # P_auxiliar2.start()


    # controladorPwm = controladorPWM()
    # controladorPwm.start(servo_fw=1200, servo_bw=1300, 
    #                      servo_suave_min=1700, servo_suave_max=500, 
    #                      servo_brusco_min=1000, servo_brusco_max=2450, tiempo=0.1)

    # vehiculoAutonomo = VehiculoAutonomo()
    # vehiculoAutonomo.comenzar()
