#DEFINICION CLASE PARA BUSCAR CIRCULOS
import numpy as np
import cv2
import ClasificadorColor
import sys

xMax = 800
xMin = 5 
yMax = 225
yMin = 50




class detectaFondo:

    def __init__(self):
        self.fondo= np.zeros((180,795,3),np.uint8)
        self.fondoHSV = np.zeros((180,795,3),np.uint8)

    def capturaFondo(self,cap,enviar1,q):
        #PREGUNTAR COCO MOSTRAR EL MENSAJE
        #"PRESIONE C PARA CAPTURAR FONDO"
        flag=0
        while(flag==0):
            frame = cap.read()
            #50:230,5:800
            frame = frame[yMin:yMax,xMin:xMax]
            enviar1.send(['N',frame])
            while not q.empty():
                variable=q.get()
                if(variable==5):
                    self.fondo = frame
                    self.fondoHSV = cv2.cvtColor(self.fondo,cv2.COLOR_BGR2HSV)
                    flag=1
                    break
                if(variable==4):
                    cap.stop()
                    exit()
                    break

    def capturaFondoAgain(self,cap):
        frame = cap.read()
        #50:230,5:800
        frame = frame[yMin:yMax,xMin:xMax]
        self.fondoHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    def getfondoHSV(self):
        return(self.fondoHSV)

class Filtrado:

    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    def filtrado(self,cap,fondo):
        #CAPTURA DEL FRAME
        self.frame = cap.read()
        #50:230,5:800
        self.frame = self.frame[yMin:yMax,xMin:xMax]
        cv2.imshow('vivo',self.frame)
        #CONVERSION A HSV
        self.frameHSV = cv2.cvtColor(self.frame,cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV',self.frameHSV)
        #RESTA EN HSV
        self.diferencia = cv2.absdiff(fondo,self.frameHSV)
        cv2.imshow('dif',self.diferencia)
        #CONVERSION A GRIS
        self.dif_gray = cv2.cvtColor(self.diferencia,cv2.COLOR_BGR2GRAY)
        #FILTRADO
        self.dif_gray = cv2.GaussianBlur(self.dif_gray,(5,5),0)
        #RELLENO
        th,im_th = cv2.threshold(self.dif_gray,25,200,cv2.THRESH_BINARY_INV)
        im_flood = im_th.copy()
        h,w = im_th.shape[:2]
        mask = np.zeros((h+2,w+2),np.uint8)
        cv2.floodFill(im_flood,mask,(0,0),255)
        im_flood_inv = cv2.bitwise_not(im_flood)
        #IMAGEN DE SALIDA
        im_out = im_th | im_flood_inv
        cv2.imshow('flood',im_out)
        #APLICACION DE FILTRO GAUSS
        gauss_fill = cv2.GaussianBlur(im_out,(3,3),0)
        #DETECCION DE BORDES
        canny = cv2.Canny(gauss_fill,25,175)
        #OPERACIONES DE DILATAMIENTO Y FILTRO NUEVAMENTE
        self.dilate = cv2.dilate(canny,self.kernel,iterations=2)
        self.dilate = cv2.GaussianBlur(self.dilate,(3,3),0)

        cv2.waitKey(5)
    def getFrameFiltrado(self):
        ret = self.dilate.copy()
        return ret
        

    def getFrameOriginal(self):
        return self.frame

class detectaCirculos:

    def __init__(self):
        self.mascaraContorno = np.zeros((yMax-yMin,xMax-xMin,3),np.uint8)
        self.mascara = np.zeros((50, 50, 3), np.uint8)
        self.pixelesPromedio = np.zeros((10, 3), np.uint8)
        self.contornos  = 0
        self.promedio = 0
        self.coefForma = 0
        self.habCircles = 0
        self.circuloEncontrado = 0
        self.circuloFalling = 0
        self.CantObjetos = 0
        self.indicepic = 0
        self.objetoNoCircular = 0

    def contornoExterior(self,dilate,frame):
        if (dilate is not None):
            cv2.imshow('dilate',dilate)
            cv2.waitKey(1)
            _, contornos, _ = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            for cnt in contornos:

                (x,y,w,h) = cv2.boundingRect(cnt)
                w = float(w)
                h = float(h)
                
                if(w*h >= 200):
                    self.coefForma =  float (w)/float(h)
                    #if (self.coefForma > 0.1):
                    #self.CantObjetos = self.CantObjetos + 1



                    if ((self.coefForma)> 0.85 and  (self.coefForma) <1.15):
                        print ("COEFICIENTE: ",float (w)/float(h))
                        '''
                        COMENTO ESTO PORQE CREOOO Q NO SIRVE PARA NADA
                        cv2.rectangle(frame,(x,y),(x+w,y+h),[255,255,255],3)
                        mascara = frame[y:y+h,x:x+w]
                        mascara_dilate = dilate.copy()
                        '''
                        self.mascaraContorno = np.zeros((yMax-yMin,xMax-xMin,3),np.uint8)
                        cv2.drawContours(self.mascaraContorno,cnt,-1,(255,255,0),2)
                        cv2.imshow('contorno',self.mascaraContorno)
                        cv2.waitKey(1)
                        self.habCircles = 1

                else:
                    self.objetoNoCircular=1

    def buscaCirculos(self,mascaraDilate,frame):

        if(self.habCircles == 1):

            self.habCircles = 0

            self.circles = cv2.HoughCircles(mascaraDilate,cv2.HOUGH_GRADIENT,3,250,
                                           param1=110,
                                           param2=150, #PARAMETRO EN 150 POR DEFECTO!
                                           minRadius=40, #ORIGINAL 25
                                           maxRadius=65)

            #if (self.circles is None):


            #el
            if (self.circles is not None):

                #self.CantObjetos = 0

                for circuloActual in self.circles[0,:]:

                    centroX = circuloActual[0]
                    centroY = circuloActual[1]
                    radio   = circuloActual[2]

                    if ((centroX>= 30 and centroX <= 770) and (centroY>= 30 and centroY<= 800)):
                        self.circuloEncontrado = 1
                        #cv2.circle(frame, (centroX,centroY), radio,[0,255,0], 5)
                        cv2.imshow('circulo',frame)
                        #cv2.imwrite('pic{:>05}.jpg'.format(self.indicepic),frame)
                        cv2.waitKey(1)

                        #self.indicepic = self.indicepic+1
                        mascara = frame[int(centroY)-25:int(centroY)+25,int(centroX)-25:int(centroX)+25]

                        if mascara.size:
                            color_promedio_columnas = np.average(mascara,axis = 0)
                            self.promedio = np.average(color_promedio_columnas, axis = 0)
                    else:
                        self.circuloEncontrado = 0
                        #self.objetoNoCircular = 1

            else:
                self.objetoNoCircular = 1




    def CirculoDetectado(self):
        #Devuelve 1 en el caso de que se haya detectado un circulo
        if (self.circuloEncontrado == 1):
            self.circuloEncontrado = 0
            return 1

        return 0

    def ObjetoDetectado(self):
        #Devuelve 1 en el caso de qe se haya detectado un objeto

        if(self.objetoNoCircular == 1):
            self.objetoNoCircular = 0
            return 1

        #if ((self.CantObjetos > 12) and (self.circuloEncontrado is 0)):
            #self.CantObjetos = 0
            #return 1

        return 0

    def getPromedio(self):
        #devuelvo color promedio de la tapita detectada
        return self.promedio

class Promedio:

    def __init__(self):
        self.pixelesPromedio = np.zeros((10,3),np.uint8)
        self.promedio = np.zeros((1, 3), np.uint8)
        self.indice = 0
        self.circuloFalling = 0
        self.incrementarTiempo = 0
        self.Ready = 0

    def incrementaTiempo(self):
        if (self.incrementarTiempo == 1):
            self.circuloFalling = self.circuloFalling + 1
        if (self.circuloFalling > 10):

            self.Ready = 0
            self.circuloFalling = 0
            self.incrementarTiempo = 0
            self.indice = 0


    def agregaPixel(self,pixel):
        if (self.Ready == 0):
            self.incrementarTiempo = 1
            self.pixelesPromedio[self.indice] = pixel
            self.indice = self.indice + 1


    def getPixelPromedio(self):

            #Calculo del color promedio almacenado durante la caida de la tapa
            self.promedio[0, 0] = np.uint8(np.mean(self.pixelesPromedio[0:self.indice, 0]))
            self.promedio[0, 1] = np.uint8(np.mean(self.pixelesPromedio[0:self.indice, 1]))
            self.promedio[0, 2] = np.uint8(np.mean(self.pixelesPromedio[0:self.indice, 2]))

            #Reinicio el contador de tiempo y el indice
            self.indice = 0
            #self.incrementarTiempo = 0
            self.circuloFalling = 0

            #Devuelvo color promedio
            return self.promedio

    def getPromedioReady(self):
        if(self.Ready == 0):
            if ((self.circuloFalling >= 10) or (self.indice >= 1)):
                self.Ready = 1
                #print 'Ready'
                return 1
        return 0
