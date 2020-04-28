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
            self.depositoHallado = 'null'
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
            self.ultima_distancia_arr = np.zeros(20)
            self.indice_ultima_posicion_3 = 0
            #Luminosidad Ambiente
            self.multiplicadorLuminosidadAmbiente = 2
            self.indiceCircular = 0
            self.arrayCircular = np.zeros(5)

            self.estuveCruzandoBocacalle = False
            self.contandoFramesCruzando = 0
            self.tiempoParaCruzarInicial = 0
            self.siguiendoLineaSuperior = False
            #Triangulos
            self.XtrianguloSuperior = []
            self.YtrianguloSuperior = []
            # self.XtrianguloInferior = []
            self.ultimaDiagonalAmarilla = True

            self.esperarHastaObjetoDetectado = False
            self.cantidad_veces_detectado_0 = 0
            self.cantidad_veces_detectado_1 = 0

            self.contandoFramesDeteccionObjetos = 0
            self.mediana_y = 0
            self.contandoFramesEstandoTorcido = 0

            self.paseElSemaforo = False
            
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
                for barcode in barcodes:
                    barcodeData = barcode.data.decode("utf-8")
                    if barcodeData[0] == 'I':
                        cv2.destroyWindow('buscandoQR')
                        print("Deposito a buscar: ", barcodeData[1])
                        self.depositoABuscar = barcodeData[1]
                        self.tiempoDeEsperaInicial = time.time()

        def _buscar_qr(self, frame):
            barcodes = pyzbar.decode(frame)
            if barcodes:
                qr_encontrado = barcodes[0].data.decode("utf-8")
                if len(qr_encontrado) == 2:
                    if qr_encontrado[0] == 'F' and qr_encontrado[1] == self.depositoABuscar and not self.listoParaReiniciar:
                        print('Dejando paquete!!')
                        self.depositoHallado = 'null'
                        enviar1.send('stopAndIgnore')
                        self.listoParaReiniciar = True
                elif len(qr_encontrado) == 1:
                    if qr_encontrado[0] == 'P' and self.listoParaReiniciar == True:
                        print("inicio hallado")
                        enviar1.send('stopAndIgnore')
                        self.listoParaReiniciar = False
                        self.tiempoDeEsperaInicial = -1
                        self.depositoABuscar = -1

        def _detectarRojo(self,frame):
            #Defino parametros HSV para detectar color rojo 
            lower_red = np.array([0, 10, 40])
            upper_red = np.array([10, 100, 100])
            #Aplico filtro de color con los parametros ya definidos
            hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
            # y, x = np.where(mask_red == 255)
            # yindice = np.where(abs(y-280) < 280*0.3)
            #Proceso
            # if len(x[yindice]) > 700:

            y, x = np.where(mask_red == 255)
            if len(x) == 0:
                return False
            self.mediana_y = int(statistics.median_low(y))
            # print(self.mediana_y)

            # if len(x) > 1000:
            if (280 < self.mediana_y) and (len(x)>500):
                return True
            else:
                return False

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

        def _detectarLineaVerde(self, frameOriginal):
            #Corto el frame
            frame = frameOriginal[320:480,0:int(self.width)] #320,480
            #Defino parametros HSV para detectar color verde 
            lower_green = np.array([40, int(20*self.multiplicadorLuminosidadAmbiente), 100])
            upper_green = np.array([80, 230, 140])
            #Aplico filtro de color con los parametros ya definidos
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(hsv_green, lower_green, upper_green)

            # lower_green = np.array([50, 80, 20])
            # upper_green = np.array([75, 120, 60])
            
            # #Aplico filtro de color con los parametros ya definidos
            # frame = cv2.GaussianBlur(frame, (3, 3), 0)
            # hls_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            # mask_green = cv2.inRange(hls_green, lower_green, upper_green)

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
            # distancia_al_centro = (self.width/2) - self.ubicacion_punto_verde
            # if x.size > 50 and distancia_al_centro < 320:    
            #     m,b = np.polyfit(x,y,1)
            #     print('pendiente:',m)
            #     if m < 0:
            #         self.girandoHaciaDerecha = True
            #     else:
            #         self.girandoHaciaDerecha = False
            # #     print('##########################' ,m,b)
            #     cv2.line(mask_green,(int(-b/m),0),(int((160-b)/m),160),(255,255,255), 3)
            # #     real_m = 160/(((160-b)/m)+b/m)
            # #     print(real_m)

            cv2.line(mask_green,(self.ubicacion_punto_verde,0),(self.ubicacion_punto_verde,480),(255,255,255), 2)
            # cv2.imshow('FiltroVerde', mask_green)
        
        def _tomarDecisionMovimiento(self):
            # Si detecto la bocacalle me preparo para doblar o seguir, esta bandera se limpia sola cuando terminamos de cruzar
            if not self.esperarHastaObjetoDetectado:
                if self.bocacalleDetectada:
                    if self.listoParaReiniciar:
                        self._moverVehiculoCruzarBocacalle()
                    elif (self.depositoHallado == self.depositoABuscar) and (self.depositoHallado != 'null'):
                        self._moverVehiculoEnLineaVerde() #Doblar
                    elif (self.depositoHallado != self.depositoABuscar) and (self.depositoHallado != 'null'):
                        # print('########################################################################CRUZAR')
                        self._moverVehiculoCruzarBocacalle() #Seguir derecho

                # Si no detecto bocacalle estoy girando en la linea verde nuevamente o recien comenzando el programa
                else:
                    self._moverVehiculoEnLineaVerde() #Sigue derecho
            
        def _moverVehiculoEnLineaVerde(self):
            # print('entrando a la funcion de mover en linea verde')
            if not self.siguiendoLineaSuperior:
                distancia_al_centro = (self.width/2) - self.ubicacion_punto_verde
                self.contandoFramesCruzando = 0
                # print('NORMAL SIN BOCACALLE')
            else:
                # print('verificando condicion')
                if (time.time() - self.tiempoParaCruzarInicial) <2.5:
                    # print('cruzando a ciegas')
                    return
                distancia_al_centro_inferior = (self.width/2) - self.ubicacion_punto_verde
                if abs(distancia_al_centro_inferior) < 100:
                    self.contandoFramesCruzando += 1
                    print('frames cruzando:', self.contandoFramesCruzando)
                if self.contandoFramesCruzando >= 5:
                    self.siguiendoLineaSuperior = False
                    print('//////////////////////////////////////// LIMPIANDO FLAG')
                    distancia_al_centro = distancia_al_centro_inferior
                else:
                    # print('viendo arriba, cuenta: ', self.contandoFramesCruzando)
                    frame = copy.deepcopy(self.frameCompleto[0:160,0:int(self.width)])
                    lower_green = np.array([40, int(20*self.multiplicadorLuminosidadAmbiente), 100])
                    upper_green = np.array([80, 230, 140])
                    frame = cv2.GaussianBlur(frame, (3, 3), 0)
                    hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
                    y, x = np.where(mask_green == 255)
                    try:
                        if len(x) < 100:
                            x_mid = 0
                        else:
                            x_mid= statistics.median(x)
                    except:
                        x_mid = 0
                    x_mid_int=int(round(x_mid))
                    ubicacion_punto_verde_superior = x_mid_int
                    distancia_al_centro = (self.width/2) - ubicacion_punto_verde_superior

            # print('verificando1')
            # print('distancia al centro: ', distancia_al_centro)
            if abs(distancia_al_centro) == 320:
                # print('verificando2')
                if self.contandoFramesParado != 3:
                    enviar1.send('stop')
                    self.contandoFramesParado += 1
                    self.contandoFramesBackward = 0
                else:
                    # print('/////////////////////////////////////////////////////////     ', self.ultima_distancia)
                    if self.ultima_distancia <= -250:
                        enviar1.send('giroBruDer')
                    elif self.ultima_distancia >= 250:
                        enviar1.send('giroBruIzq')
                    else:
                        enviar1.send('backward')
                    self.contandoFramesBackward += 1
                    if self.contandoFramesBackward == 2:
                        self.contandoFramesParado = 0
            
            else:
                # print('verificando3', distancia_al_centro)
                limite1 = 40
                limite2 = 170
                distancia_al_centro -= 11
                self.contandoFramesParado = 0
                if distancia_al_centro > limite1 and abs(distancia_al_centro) < limite2:
                    enviar1.send('giroSuaIzq')
                elif distancia_al_centro < -limite1 and abs(distancia_al_centro) < limite2:
                    enviar1.send('giroSuaDer')
                elif 320 > abs(distancia_al_centro) >= limite2 and self.ultima_distancia <= 0:
                        enviar1.send('giroBruDer')
                elif 320 > abs(distancia_al_centro) >= limite2 and self.ultima_distancia > 0:
                        enviar1.send('giroBruIzq')
                else:
                    enviar1.send('forward')

            if abs(distancia_al_centro) < 320:
                if (self.indice_ultima_posicion_3 is 20): #Resetea el indice del buffer circular
                    self.indice_ultima_posicion_3 = 0
                self.ultima_distancia_arr[self.indice_ultima_posicion_3] = distancia_al_centro #Agrega los valores al buffer circular
                self.ultima_distancia = int(statistics.median(self.ultima_distancia_arr)) 
                self. indice_ultima_posicion_3 += 1

            self.estuveCruzandoBocacalle = False

        def _moverVehiculoCruzarBocacalle(self):
            if not self.estuveCruzandoBocacalle:
                Thread(target=self._hiloParaCruzarBocacalle, args=()).start()
                self.tiempoParaCruzarInicial = time.time()
            elif (time.time()-self.tiempoParaCruzarInicial > 2.5):
                Thread(target=self._hiloParaCruzarBocacalle, args=()).start()
                self.tiempoParaCruzarInicial = time.time()
            self.siguiendoLineaSuperior = True
            self.estuveCruzandoBocacalle = True

        def _hiloParaCruzarBocacalle(self):
            #Obtiene ubicacion del punto
            tiempoParaCruzarInicial = time.time()
            print('lanzando hilo')
            contandoFramesParado = 0
            contandoFramesBackward = 0

            while ((time.time()-tiempoParaCruzarInicial) < 2):

                frame = copy.deepcopy(self.frameCompleto[0:240,0:int(self.width)])
                lower_green = np.array([40, int(20*self.multiplicadorLuminosidadAmbiente), 100])
                upper_green = np.array([80, 230, 140])
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
                y, x = np.where(mask_green == 255)
                cantidad_puntos = round(len(x)*0.2)
                try:
                    indices = np.argpartition(y, cantidad_puntos)
                    punto_a_seguir = statistics.median(x[indices[:cantidad_puntos]])
                except:
                    punto_a_seguir = 0

                distancia_al_centro = (self.width/2) - punto_a_seguir
                # print(distancia_al_centro)

                # try:
                #     indices = np.argpartition(self.YtrianguloSuperior, 300)
                #     punto_a_seguir = statistics.median(self.XtrianguloSuperior[indices[:300]])
                # except:
                #     try:
                #         punto_a_seguir = statistics.median(self.XtrianguloSuperior)
                #     except:
                #         punto_a_seguir = 0
                
                # distancia_al_centro =  (self.width/2) -  punto_a_seguir
                #Se mueve siguiendo ese punto
                if abs(distancia_al_centro) == 320:
                    # print('verificando2')
                    if contandoFramesParado != 3:
                        enviar1.send('stop')
                        contandoFramesParado += 1
                        contandoFramesBackward = 0
                    else:
                        enviar1.send('backward')
                        contandoFramesBackward += 1
                        if contandoFramesBackward == 2:
                            contandoFramesParado = 0
                else:
                    limite1 = 40
                    limite2 = 170
                    if limite2 > distancia_al_centro > limite1:
                        enviar1.send('giroSuaIzq')
                    elif -limite1 > distancia_al_centro > -limite2:
                        enviar1.send('giroSuaDer')
                    elif distancia_al_centro >= limite2:
                        enviar1.send('giroBruIzq')
                    elif -limite2 >= distancia_al_centro:
                        enviar1.send('giroBruDer')
                    else:
                        enviar1.send('forward')
            print('fin hilo')

        def _buscarDeposito(self):
            self._buscar_qr(self.frameCompleto)

        def _actualizarValorSaturacion(self):
            self.arrayCircular[self.indiceCircular] = self._obtenerLuminosidadAmbiente(self.frameCompleto)
            self.indiceCircular += 1
            if self.indiceCircular == 5:
                self.indiceCircular = 0
            self.multiplicadorLuminosidadAmbiente = np.mean(self.arrayCircular)

        def _detectarBocacalleVerde(self):
            frame = copy.deepcopy(self.frameCompleto)
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
                puntos_arriba = 1500
                puntos_abajo = 2000
                if i==0:
                    # Chequeo diagonal amarilla
                    upper_left_triangle = np.flipud(np.tril(np.flipud(mask_green), 0)) # Upper triangle of an array
                    lower_right_triangle = np.fliplr(np.tril(np.fliplr(mask_green), -1)) # Lower triangle of an array
                    y_up_left, x_up_left = np.where(upper_left_triangle == 1)
                    y_down_right, x_down_right = np.where(lower_right_triangle == 1)
                    if self.ultimaDiagonalAmarilla:
                        self.XtrianguloSuperior = x_up_left
                        self.YtrianguloSuperior = y_up_left
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
                        # self.XtrianguloSuperior = x_up_left    
                        # self.XtrianguloInferior = x_down_right       
                        self.ultimaDiagonalAmarilla = True
                        self.bocacalleDetectada = True
                        break

                else:
                    # Chequeo diagonal azul
                    lower_left_triangle = np.tril(mask_green, -1) # Lower triangle of an array
                    upper_right_triangle = np.fliplr(np.flipud(np.tril(np.flipud(np.fliplr(mask_green)), 0))) # Upper triangle of an array
                    y_up_right, x_up_right = np.where(upper_right_triangle == 1)
                    y_down_left, x_down_left = np.where(lower_left_triangle == 1)
                    if not self.ultimaDiagonalAmarilla:
                        self.XtrianguloSuperior = x_up_right
                        self.YtrianguloSuperior = y_up_right
                    if len(x_up_right) > puntos_arriba and len(x_down_left) > puntos_abajo:
                        suficientesPuntos = True
                    else:
                        self.bocacalleDetectada = False
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
                    except:
                        diagonalNoCruza = False

                    if suficientesPuntos and diagonalNoCruza:  
                        # self.XtrianguloSuperior = x_up_right
                        # self.XtrianguloInferior = x_down_left
                        self.ultimaDiagonalAmarilla = False         
                        self.bocacalleDetectada = True
                    else:
                        self.bocacalleDetectada = False

        def comenzar(self):
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
                    # out.write(frameCompleto)
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
                        if not self.cartelDetectado:
                            if self._detectarRojo(frameCompleto): #ToDO: Falta hacer que solo busque cuando ve la senda por primera vez
                                #cuando deja de ver rojo deja de buscar carteles, hay que hacer una bandera 
                                print('##################### rojo detectado')
                                enviar1.send('stopPrioritario')
                                self.esperarHastaObjetoDetectado = True
                                # para dejarla levantada y mientras este levantada va a buscar carteles. una vez que encuentra un cartel se limpia
                                
                                # if self.estoyDistanciaCorrecta:
                                self.contandoFramesDeteccionObjetos += 1
                                if self.contandoFramesDeteccionObjetos == 5:
                                    self.contandoFramesDeteccionObjetos = 0
                                    class_ids = self._buscarObjetos(frameCompleto)
                                    print('Objetos detectados: ', class_ids) # ToDo: Borrar print

                                    if class_ids:
                                        if not self.paseElSemaforo:
                                            if ((1 in class_ids) or (0 in class_ids)) and (len(class_ids) == 1):
                                                pass
                                            elif ((2 in class_ids) or (3 in class_ids)) and (len(class_ids) == 1):
                                                pass
                                            elif ((0 in class_ids) or (1 in class_ids)) and ((2 in class_ids) or (3 in class_ids)) and (len(class_ids) == 2):
                                                pass
                                        
                                        else:
                                            if (0 in class_ids) and (len(class_ids) == 1):
                                                pass
                                            elif (1 in class_ids) and (len(class_ids) == 1):
                                                pass


                                        if ((1 in class_ids) or (0 in class_ids)) and (len(class_ids) == 1):
                                            self.contandoFramesEstandoTorcido += 1
                                            if self.contandoFramesEstandoTorcido == 10:
                                                self.contandoFramesEstandoTorcido = 0
                                                enviar1.send('giroEnElLugarDer')
                                        elif (3 in class_ids) and (len(class_ids) == 1):
                                            self.contandoFramesEstandoTorcido += 1
                                            if self.contandoFramesEstandoTorcido == 10:
                                                self.contandoFramesEstandoTorcido = 0
                                                enviar1.send('giroEnElLugarIzq')
                                        elif (2 in class_ids):
                                            self.contandoFramesEstandoTorcido = 0
                                            tiempoInicialLuegoDeDeteccionCartel = time.time()
                                            self.cantidad_veces_detectado_1 += 1
                                            self.cantidad_veces_detectado_0 = 0
                                            # self.RojoDetectado = False
                                            print('##################### cartel uno detectado')
                                            if (len(class_ids) == 1):
                                                if self.cantidad_veces_detectado_1 >= 3:
                                                    # self.cantidad_veces_detectado_0 = 0
                                                    self.cantidad_veces_detectado_1 = 0 
                                                    
                                                    self.cartelDetectado = True
                                                    self.depositoHallado = str(2)
                                                    self.esperarHastaObjetoDetectado = False
                                        elif (3 in class_ids):
                                            self.contandoFramesEstandoTorcido = 0
                                            tiempoInicialLuegoDeDeteccionCartel = time.time()
                                            self.cantidad_veces_detectado_0 += 1
                                            self.cantidad_veces_detectado_1 = 0
                                            # self.RojoDetectado = False
                                            print('##################### cartel cero detectado')
                                            if ((0 in class_ids) or (1 in class_ids)) and (len(class_ids) == 2):
                                                if 1 in class_ids and self.cantidad_veces_detectado_0 >= 3:
                                                    self.cantidad_veces_detectado_0 = 0
                                                    self.cartelDetectado = True
                                                    self.depositoHallado = str(1)
                                                    self.esperarHastaObjetoDetectado = False
                                                    print('##################### cartel cero detectado y semaforo verde')
                                    else:
                                        self.contandoFramesEstandoTorcido += 1
                                        if self.contandoFramesEstandoTorcido == 10:
                                                self.contandoFramesEstandoTorcido = 0
                                                enviar1.send('giroEnElLugarDer')
                                        # Si no veo nada
                                # elif not self.hiloCentradoRojoYaLanzado:
                                #     Thread(target=self._hiloParaAcercarseSendaRoja, args=()).start()
                        else:
                            self.contandoFramesEstandoTorcido = 0
                            if not self._detectarRojo(frameCompleto):
                                # Espero 10segundos para borrar la bandera de cartel detectado
                                if (time.time()-tiempoInicialLuegoDeDeteccionCartel) > 3:
                                    self.cartelDetectado = False

                        Thread(target=self._actualizarValorSaturacion, args=()).start()
                        # print("FPS 1: ", (1/(time.time()-tiempoInicialFPS)))
                        # Busco constantemente el punto central de la linea verde
                        self._detectarLineaVerde(frameCompleto)
                        # print("FPS 2: ", (1/(time.time()-tiempoInicialFPS)))
                        # Busco constantemente la bocacalle y su fin
                        Thread(target=self._detectarBocacalleVerde, args=()).start()
                        # print("FPS 3: ", (1/(time.time()-tiempoInicialFPS)))
                        # En base a los resultados de self._detectarBocacalle() decido si seguir la linea verde o cruzar la bocacalle
                        self._tomarDecisionMovimiento()
                        # print("FPS 4: ", (1/(time.time()-tiempoInicialFPS)))
                        Thread(target=self._buscarDeposito, args=()).start()

                        # Display the resulting frame
                        cv2.imshow('frameCompleto', self.frameCompleto[0][0])
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
            # out.release()
            # out2.release()
            # Closes all the frames
            cv2.destroyAllWindows()
            exit()

    vehiculoAutonomo = VehiculoAutonomo()
    vehiculoAutonomo.comenzar()


if __name__ == "__main__":
    # out = cv2.VideoWriter('outputGirandoPistaUnoOpt2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
    # out2 = cv2.VideoWriter('outputGirandoPistaUnoOpt3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,160))
    # out = cv2.VideoWriter('video_de_prueba.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))

    enviar1, recibir1 = Pipe()
    
    P_principal = Process(target=procesoPrincipal, args=(enviar1,))
    P_auxiliar = Process(target=procesoAuxiliar, args=(recibir1,))
    
    P_principal.start()
    P_auxiliar.start()