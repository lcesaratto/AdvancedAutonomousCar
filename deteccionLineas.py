import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
from pyzbar import pyzbar
import copy
from controlPWM import procesoAuxiliar
from multiprocessing import Process, Pipe
from threading import Thread, Event


def procesoPrincipal(enviar1):

    class VehiculoAutonomo (object):
        def __init__(self):
            self.e = Event()
            # self.contandoFramesNico = 0
            # self.contandoFramesGonzalo = 0
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

            self.buscandoParadaEnDeposito = False

            self.buscandoParadaEnInicio = False

            self.arrayParametrosX = [0,0]

            self.m_corr, self.b_corr = self._cargarFactoresAlineacion()

            self.tiempoInicialLuegoDeDeteccionCartel = 0

            self.noLanceHiloParaFalsoRojo = True

            # self.NuevoFrameParaProcesar = False
            
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

        def _cargarFactoresAlineacion(self):
            y_corr = np.loadtxt('data/m_vista1.out')
            x_corr = np.loadtxt('data/x_abajo1.out')
            m_corr,b_corr = np.polyfit(x_corr,y_corr,1)
            return m_corr, b_corr

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
            # cv2.putText(frame, str(self.contandoFramesGonzalo), (500,400), self.font, 2, (0,0,255), 3)
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
                        self.buscandoParadaEnDeposito = True
                        Thread(target=self._hiloDetectarRojoEnDeposito, args=(self.e,)).start()
                        self.listoParaReiniciar = True
                        # self.buscandoRojoEnDeposito = True
                    elif qr_encontrado[0] == 'F' and self.listoParaReiniciar and qr_encontrado[1] != self.depositoABuscar and self.noLanceHiloParaFalsoRojo:
                        print('detecto falso qr')
                        self.noLanceHiloParaFalsoRojo = False
                        self.buscandoParadaEnDeposito = True
                        Thread(target=self._hiloDetectarFalsoRojo, args=(self.e,)).start()
                elif len(qr_encontrado) == 1:
                    if qr_encontrado[0] == 'P' and self.listoParaReiniciar == True:
                        print("inicio hallado")
                        self.buscandoParadaEnInicio = True
                        Thread(target=self._hiloDetectarRojoEnInicio, args=(self.e,)).start()
                        self.listoParaReiniciar = False

        def _detectarRojo(self,frame):  
            #Defino parametros HSV para detectar color rojo
            if self.buscandoParadaEnDeposito or self.buscandoParadaEnInicio:
                return False

            lower_red = np.array([0, 10, 40])
            upper_red = np.array([10, 100, 100])
            #Aplico filtro de color con los parametros ya definidos
            hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            mask_red = cv2.inRange(hsv_red, lower_red, upper_red)

            y, x = np.where(mask_red == 255)
            if len(x) == 0:
                return False
            self.mediana_y = int(statistics.median_low(y))
  
            if (280 < self.mediana_y) and (len(x)>500):
                return True
            else:
                return False

        def _hiloDetectarRojoEnInicio(self, event):

            def alinear():
                ultimo_m_recibido = 0
                while True:
                    [m, b] = self.arrayParametrosX
                    if m == ultimo_m_recibido:
                        continue
                    else:
                        ultimo_m_recibido = m
                    if m != 100:
                        x_ab = m*480+b
                        m_tabla = x_ab*self.m_corr+self.b_corr
                        if ( (m_tabla-0.1) < m < (m_tabla+0.1) ):
                            print('OK')
                            break
                        else:
                            if m < m_tabla:
                                print("GIRAR A LA DERECHA")
                                enviar1.send('giroEnElLugarDer')
                                time.sleep(0.5)
                            else:
                                print("GIRAR A LA IZQUIERDA")
                                enviar1.send('giroEnElLugarIzq')
                                time.sleep(0.5)

            def avanzar(hastaMitadCamino = False):
                try:
                    frame = copy.deepcopy(self.frameCompleto)
                    lower_red = np.array([0, 10, 40])
                    upper_red = np.array([10, 100, 100])
                    #Aplico filtro de color con los parametros ya definidos
                    hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
                    mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
                    y, x = np.where(mask_red == 255)
                    mediana_y = int(statistics.median_low(y))
                    segundos = round(-0.00286 * mediana_y + 1.84, 1)
                    if hastaMitadCamino:
                        segundos = round(segundos / 2, 1)
                except:
                    print('PERDIO LA LINEA ROJA EN INICIO')
                    segundos = 0.2
                enviar1.send(('forwardPersonalizado_'+ str(segundos)))

            while True:
                if not self.buscandoParadaEnDeposito:
                    event.wait()
                    frame = copy.deepcopy(self.frameCompleto)

                    lower_red = np.array([0, 10, 40])
                    upper_red = np.array([10, 100, 100])
                    #Aplico filtro de color con los parametros ya definidos
                    hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
                    mask_red = cv2.inRange(hsv_red, lower_red, upper_red)

                    y, x = np.where(mask_red == 255)
                    if len(x) == 0:
                        event.clear()
                        continue
                    mediana_y = int(statistics.median_low(y))

                    if (200 < mediana_y) and (len(x)>100):
                        self.esperarHastaObjetoDetectado = True
                        enviar1.send('stopPrioritario')
                        event.clear()
                        break
                event.clear()

            time.sleep(1)
            if (200 < mediana_y < 300):
                for i in range(2):
                    # endereza
                    alinear()
                    time.sleep(1)
                    #avance a ciegas
                    if i == 0:
                        avanzar(True)
                    else:
                        avanzar()
                    time.sleep(1)
            else:
                # endereza
                alinear()
                time.sleep(1)
                #avance a ciegas
                avanzar()
                time.sleep(1)
            self.tiempoDeEsperaInicial = -1
            self.depositoABuscar = -1
            self.esperarHastaObjetoDetectado = False
            self.buscandoParadaEnInicio = False
            self.paseElSemaforo = False

        def _hiloDetectarRojoEnDeposito(self, event):
            if self.buscandoParadaEnDeposito:
                while True:
                    event.wait()
                    # self.lock.adquire()
                    frame = copy.deepcopy(self.frameCompleto)

                    lower_red = np.array([0, 10, 40])
                    upper_red = np.array([10, 100, 100])
                    #Aplico filtro de color con los parametros ya definidos
                    hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
                    mask_red = cv2.inRange(hsv_red, lower_red, upper_red)

                    y, x = np.where(mask_red == 255)
                    if len(x) == 0:
                        event.clear()
                        continue
                    mediana_y = int(statistics.median_low(y))

                    if (200 < mediana_y) and (len(x)>200):
                        # return True
                        self.esperarHastaObjetoDetectado = True
                        enviar1.send('stopPrioritario')
                        event.clear()
                        break
                    event.clear()

                time.sleep(1)
                # endereza
                ultimo_m_recibido = 0
                while True:
                    [m, b] = self.arrayParametrosX
                    if m == ultimo_m_recibido:
                        continue
                    else:
                        ultimo_m_recibido = m
                    if m != 100:
                        x_ab = m*480+b
                        m_tabla = x_ab*self.m_corr+self.b_corr
                        if ( (m_tabla-0.1) < m < (m_tabla+0.1) ):
                            print('OK')
                            break
                        else:
                            if m < m_tabla:
                                print("GIRAR A LA DERECHA")
                                enviar1.send('giroEnElLugarDer')
                                time.sleep(0.5)
                            else:
                                print("GIRAR A LA IZQUIERDA")
                                enviar1.send('giroEnElLugarIzq')
                                time.sleep(0.5)

                time.sleep(1)
                #avance a ciegas
                try:
                    frame = copy.deepcopy(self.frameCompleto)
                    lower_red = np.array([0, 10, 40])
                    upper_red = np.array([10, 100, 100])
                    #Aplico filtro de color con los parametros ya definidos
                    hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
                    mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
                    y, x = np.where(mask_red == 255)
                    mediana_y = int(statistics.median_low(y))
                    segundos = round(-0.00286 * mediana_y + 1.84, 1)
                except:
                    print('PERDIO LA LINEA ROJA EN DEPOSITO')
                    segundos = 0.2
                enviar1.send(('forwardPersonalizado_'+ str(segundos)))
                time.sleep(5)
                self.esperarHastaObjetoDetectado = False
                time.sleep(5)
                self.buscandoParadaEnDeposito = False

        def _hiloDetectarFalsoRojo(self, event):
            estoyDetectandoRojo = False
            tiempoDesdeUltimoRojoDetectado = -1

            while True:
                event.wait()
                frame = copy.deepcopy(self.frameCompleto)

                lower_red = np.array([0, 10, 40])
                upper_red = np.array([10, 100, 100])
                #Aplico filtro de color con los parametros ya definidos
                hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
                mask_red = cv2.inRange(hsv_red, lower_red, upper_red)

                y, x = np.where(mask_red == 255)
                if len(x) < 200:
                    if estoyDetectandoRojo:
                        tiempoDesdeUltimoRojoDetectado = time.time()
                        estoyDetectandoRojo = False
                elif len(x) >= 200:
                    tiempoDesdeUltimoRojoDetectado = -1
                    estoyDetectandoRojo = True

                if (time.time() - tiempoDesdeUltimoRojoDetectado) > 3 and tiempoDesdeUltimoRojoDetectado != -1:
                    event.clear()
                    break
                event.clear()

            self.noLanceHiloParaFalsoRojo = True
            self.buscandoParadaEnDeposito = False

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
                    cv2.rectangle(self.frameCompleto, (x, y), (x + w, y + h), color, 2)
                    # cv2.putText(frameMostrado, label, (x, y + 30), self.font, 3, color, 2)
                    cv2.putText(self.frameCompleto, label, (x-30, y + 60), self.font, 2, color, 3)
                    cv2.putText(self.frameCompleto, 'Conf: ' + (str(confidence)[:4]), (x-30, y + 30), self.font, 2, color, 3)

                # cv2.imshow('buscandoObjetos',frameMostrado)
                
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
            distancia_al_centro = (self.width/2) - self.ubicacion_punto_verde
            y += 320
            y_indice = np.where(y>400)
            x_nuevo = x[y_indice]
            y_nuevo = y[y_indice]
            if x_nuevo.size > 50 and distancia_al_centro < 320:
                m,b = np.polyfit(y_nuevo,x_nuevo,1)
                m2 = 1/m
                b2 = -b/m
                #cv2.line(self.frameCompleto, (int((320-b2)/m2),320), (int((480-b2)/m2),480),(255,0,0), 3)

                self.arrayParametrosX = [m,b]
            else:
                self.arrayParametrosX = [100,100]

        def _tomarDecisionMovimiento(self):
            # Si detecto la bocacalle me preparo para doblar o seguir, esta bandera se limpia sola cuando terminamos de cruzar
            if not self.esperarHastaObjetoDetectado:
                if self.bocacalleDetectada:
                    if self.listoParaReiniciar:
                        self._moverVehiculoCruzarBocacalle()
                    elif (self.depositoHallado == self.depositoABuscar) and (self.depositoHallado != 'null'):
                        self._moverVehiculoEnLineaVerde() #Doblar
                    elif (self.depositoHallado != self.depositoABuscar) and (self.depositoHallado != 'null'):
                        self._moverVehiculoCruzarBocacalle() #Seguir derecho

                # Si no detecto bocacalle estoy girando en la linea verde nuevamente o recien comenzando el programa
                else:
                    self._moverVehiculoEnLineaVerde() #Sigue derecho
            
        def _moverVehiculoEnLineaVerde(self):
            # print('entrando a la funcion de mover en linea verde')
            if not self.siguiendoLineaSuperior:
                distancia_al_centro = (self.width/2) - self.ubicacion_punto_verde
                self.contandoFramesCruzando = 0
            else:
                if (time.time() - self.tiempoParaCruzarInicial) <2.5:
                    return
                distancia_al_centro_inferior = (self.width/2) - self.ubicacion_punto_verde
                if abs(distancia_al_centro_inferior) < 100:
                    self.contandoFramesCruzando += 1
                if self.contandoFramesCruzando >= 5:
                    self.siguiendoLineaSuperior = False
                    print('//////////////////////////////////////// LIMPIANDO FLAG')
                    distancia_al_centro = distancia_al_centro_inferior
                else:
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

            # print('///////////////////////////// ', distancia_al_centro)
            if abs(distancia_al_centro) == 320:
                if self.contandoFramesParado != 3:
                    enviar1.send('stop')
                    self.contandoFramesParado += 1
                    self.contandoFramesBackward = 0
                else:
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
                limite1 = 40
                limite2 = 170
                distancia_al_centro -= 11
                self.contandoFramesParado = 0
                if distancia_al_centro > limite1 and abs(distancia_al_centro) < limite2:
                    enviar1.send('giroSuaIzq')
                elif distancia_al_centro < -limite1 and abs(distancia_al_centro) < limite2:
                    enviar1.send('giroSuaDer')
                #elif 320 > abs(distancia_al_centro) >= limite2 and self.ultima_distancia <= 0:
                elif -320 < distancia_al_centro <= -limite2:
                        enviar1.send('giroBruDer')
                #elif 320 > abs(distancia_al_centro) >= limite2 and self.ultima_distancia > 0:
                elif 320 > distancia_al_centro >= limite2:
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
            # print('lanzando hilo')
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
            # print('fin hilo')

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
        
        def _tomarDesicionBasadaDeteccionObjetos(self, frameCompleto):
            # Comenzamos buscando objetos si se detecta la senda peatonal roja
            if not self.cartelDetectado and not self.listoParaReiniciar:
                if self._detectarRojo(frameCompleto): #ToDO: Falta hacer que solo busque cuando ve la senda por primera vez
                    #cuando deja de ver rojo deja de buscar carteles, hay que hacer una bandera 
                    # print('##################### rojo detectado')
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
                                    self.contandoFramesEstandoTorcido += 1
                                    if self.contandoFramesEstandoTorcido == 2:
                                        self.contandoFramesEstandoTorcido = 0
                                        enviar1.send('giroEnElLugarDer')
                                elif ((2 in class_ids) or (3 in class_ids)) and (len(class_ids) == 1):
                                    self.contandoFramesEstandoTorcido += 1
                                    if self.contandoFramesEstandoTorcido == 2:
                                        self.contandoFramesEstandoTorcido = 0
                                        enviar1.send('giroEnElLugarIzq')
                                elif ((0 in class_ids) or (1 in class_ids)) and ((2 in class_ids) or (3 in class_ids)) and (len(class_ids) == 2):
                                    if (2 in class_ids):
                                        self.contandoFramesEstandoTorcido = 0
                                        self.tiempoInicialLuegoDeDeteccionCartel = time.time()
                                        self.cantidad_veces_detectado_1 += 1
                                        self.cantidad_veces_detectado_0 = 0
                                        if 1 in class_ids and self.cantidad_veces_detectado_1 >= 3:
                                            self.cantidad_veces_detectado_1 = 0
                                            self.cartelDetectado = True
                                            self.depositoHallado = str(2)
                                            self.esperarHastaObjetoDetectado = False
                                            self.paseElSemaforo = True
                                            print('##################### cartel uno detectado y semaforo verde')
                                    elif (3 in class_ids):
                                        self.contandoFramesEstandoTorcido = 0
                                        self.tiempoInicialLuegoDeDeteccionCartel = time.time()
                                        self.cantidad_veces_detectado_0 += 1
                                        self.cantidad_veces_detectado_1 = 0
                                        if 1 in class_ids and self.cantidad_veces_detectado_0 >= 3:
                                            self.cantidad_veces_detectado_0 = 0
                                            self.cartelDetectado = True
                                            self.depositoHallado = str(1)
                                            self.esperarHastaObjetoDetectado = False
                                            self.paseElSemaforo = True
                                            print('##################### cartel cero detectado y semaforo verde')
                            
                            else:
                                if (2 in class_ids) and (len(class_ids) == 1):
                                    self.contandoFramesEstandoTorcido = 0
                                    self.tiempoInicialLuegoDeDeteccionCartel = time.time()
                                    self.cantidad_veces_detectado_1 += 1
                                    self.cantidad_veces_detectado_0 = 0
                                    if self.cantidad_veces_detectado_1 >= 3:
                                        self.cantidad_veces_detectado_1 = 0 
                                        self.cartelDetectado = True
                                        self.depositoHallado = str(2)
                                        self.esperarHastaObjetoDetectado = False
                                        print('##################### cartel uno detectado')
                                elif (3 in class_ids) and (len(class_ids) == 1):
                                    self.contandoFramesEstandoTorcido = 0
                                    self.tiempoInicialLuegoDeDeteccionCartel = time.time()
                                    self.cantidad_veces_detectado_0 += 1
                                    self.cantidad_veces_detectado_1 = 0
                                    if self.cantidad_veces_detectado_0 >= 3:
                                        self.cantidad_veces_detectado_0 = 0 
                                        self.cartelDetectado = True
                                        self.depositoHallado = str(1)
                                        self.esperarHastaObjetoDetectado = False
                                        print('##################### cartel cero detectado')

                        else:
                            if self.paseElSemaforo:
                                self.contandoFramesEstandoTorcido += 1
                                if self.contandoFramesEstandoTorcido == 2:
                                        self.contandoFramesEstandoTorcido = 0
                                        enviar1.send('giroEnElLugarDer')

            else:
                self.contandoFramesEstandoTorcido = 0
                if not self._detectarRojo(frameCompleto):
                    # Espero 3segundos para borrar la bandera de cartel detectado
                    if (time.time()-self.tiempoInicialLuegoDeDeteccionCartel) > 3:
                        self.cartelDetectado = False

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
                    self.e.set()
                    # self.NuevoFrameParaProcesar = True
                    # self.contandoFramesGonzalo += 1
                    self.frameCompleto = frameCompleto

                    # Aca corremos la funcion que busca un codigo qr en la imagen para comenzar
                    if self.listoParaReiniciar:
                        self._buscar_qr(frameCompleto)
                    else:
                        if self.depositoABuscar == -1:
                            self._leer_qr(frameCompleto)
                    
                    # Si se cumple el tiempo de espera inicial luego de leer el QR, comenzamos a movernos
                    if (time.time()-self.tiempoDeEsperaInicial) > 5 and self.tiempoDeEsperaInicial != -1:
                        tiempoInicialFPS = time.time()

                        self._tomarDesicionBasadaDeteccionObjetos(frameCompleto)

                        Thread(target=self._actualizarValorSaturacion, args=()).start()
                        # Busco constantemente el punto central de la linea verde
                        self._detectarLineaVerde(frameCompleto)
                        # Busco constantemente la bocacalle y su fin
                        Thread(target=self._detectarBocacalleVerde, args=()).start()
                        # En base a los resultados de self._detectarBocacalle() decido si seguir la linea verde o cruzar la bocacalle
                        self._tomarDecisionMovimiento()
                        Thread(target=self._buscarDeposito, args=()).start()
                        # Display the resulting frame
                        # cv2.putText(self.frameCompleto, str(self.contandoFramesGonzalo), (500,400), self.font, 2, (0,0,255), 3)
                        cv2.imshow('frameCompleto', self.frameCompleto[0][0])
                        # self.contandoFramesNico += 1
                        # cv2.imwrite('nico/Completo_'+str(self.contandoFramesNico)+'.jpg', self.frameCompleto)
                        # cv2.imwrite('nico/Mascara_'+str(self.contandoFramesNico)+'.jpg', self.mask_green)
                        # cv2.imshow('frameVerde', self.mask_green[0][0])
                        # out.write(self.frameCompleto)
                        # out1.write(self.mask_green)
                        # Press Q on keyboard to  exit
                        key = cv2.waitKey(10)
                        if key == ord('q') or key == ord('Q'):
                            enviar1.send('stop')
                            cv2.waitKey(10)
                            enviar1.send('exit')
                            break
                        # print("FPS: ", (1/(time.time()-tiempoInicialFPS)))
                else: # Break the loop
                    break

            # When everything done, release the video capture object
            self.cap.release()
            # out.release()
            # out1.release()
            # Closes all the frames
            cv2.destroyAllWindows()
            exit()

    vehiculoAutonomo = VehiculoAutonomo()
    vehiculoAutonomo.comenzar()


if __name__ == "__main__":
    # out = cv2.VideoWriter('mostrandoNico1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
    # out1 = cv2.VideoWriter('mostrandoNico2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,160))

    enviar1, recibir1 = Pipe()
    
    P_principal = Process(target=procesoPrincipal, args=(enviar1,))
    P_auxiliar = Process(target=procesoAuxiliar, args=(recibir1,))
    
    P_principal.start()
    P_auxiliar.start()