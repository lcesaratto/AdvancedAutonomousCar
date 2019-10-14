import numpy as np
import cv2

class DatosColor:
    #Esta clase agrupa los datos de colores existentes, maximo 10 colores
    def __init__(self):
        self.Colores = np.loadtxt('COLORES.txt',np.string_)
        self.indice = np.zeros(1,np.uint8)
        f, _ = self.Colores.shape
        self.indice = f-1
        self.matrixAux = np.zeros((1, 1, 3), np.uint8)
        self.matrixAux[0, 0] = [150,150,150]
        self.matrixAux = cv2.cvtColor(self.matrixAux, cv2.COLOR_BGR2LAB)
        self.L = 0
        self.A = 0
        self.B = 0
        self.menor = 0
        self.deltaE = 0
        self.menorIndice = 0
        
        #print (self.Colores)

    def AgregarColor(self):
        #Esta funcion reemplaza los acutuales colores cargados en Colores
        self.Colores = np.loadtxt('COLORES.txt',np.string_)
        self.indice = np.zeros(1,np.uint8)
        f, _ = self.Colores.shape
        self.indice = f-1

    def ClasificadorColor(self,ColorInput,rango=50):
        #ClasificadorColores utiliza la distancia euclidiana en el espacio de color LAB
        #para comparar el color ingresado con los actuales colores disponible en Colores[]
        self.menor = 0
        self.deltaE = 0
        self.menorIndice = 0
        
        #Cabiar espacio de color de ColorInput de BGR a LAB:
        self.matrixAux[0, 0] = ColorInput
        self.matrixAux = cv2.cvtColor(self.matrixAux, cv2.COLOR_BGR2LAB)
        #print (self.matrixAux)
        self.L = int(self.Colores[0, 0]) - self.matrixAux[0, 0, 0]
        self.A = int(self.Colores[0, 1]) - self.matrixAux[0, 0, 1]
        self.B = int(self.Colores[0, 2]) - self.matrixAux[0, 0, 2]
        self.menor = np.sqrt(np.square(self.L) + np.square(self.A) + np.square(self.B))

        for i in range(1, self.indice+1):

            self.L = int(self.Colores[i,0]) - self.matrixAux[0, 0, 0]
            self.A = int(self.Colores[i,1]) - self.matrixAux[0, 0, 1]
            self.B = int(self.Colores[i,2]) - self.matrixAux[0, 0, 2]
            self.deltaE = np.sqrt(np.square(self.L) + np.square(self.A) + np.square(self.B))
    
            if self.deltaE < self.menor:
                self.menorIndice = i
                self.menor = self.deltaE
        #print (self.menor)
   
        if self.menor < rango:
            ret = self.getIdentificador(self.menorIndice)
##            print ()
            coolor=self.getNombre(self.menorIndice)
            return ret,coolor
        else:
            #print 'N'
            return 'N','Nada'

    def getIndice(self):
        return self.indice

    def getIdentificador(self,i):
        return chr(int(self.Colores[i,3]))

    def getNombre(self,i):
        return self.Colores[i,4]

