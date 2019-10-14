from __future__ import division
from Tkinter import *
from Tkinter import Menu
import tkMessageBox
import ttk
import Tkinter as tk
#pip install Pillow
#sudo apt-get install python-imaging python-imaging-tk
from PIL import ImageTk
from PIL import Image
import multiprocessing as mp
import cv2

import serial
import time
import sys
import numpy as np
from decimal import Decimal
from fractions import Fraction
#pip install VideoGet
import VideoGet
import ClasificadorColor
import testCV

def captura_frame(enviar1,q):
    #SETEO DE ARDUINO
    arduino = serial.Serial('/dev/ttyACM0',baudrate=9600)

    #SETEO DE LA CAMARA
    #640 * 304 original
    cap = VideoGet.PiVideoStream((640, 304), 90)
    cap.start()
    time.sleep(2)

    #INSTANCIA DE CLASES 
    fondo = testCV.detectaFondo()

    filtros = testCV.Filtrado()

    circulos = testCV.detectaCirculos()

    promedio = testCV.Promedio()

    color = ClasificadorColor.DatosColor()

    #Ultima letra enviada:
    UltimaLetra = 'N'
    
    #CAPTURA FONDO
    fondo.capturaFondo(cap,enviar1,q)
    frame=np.zeros((1,1,3),np.uint8)
    color_detectado=' '
    #contador=0
    error=0
    #INICIO DEL PROGRAMA
    while(1):
        
        promedio.incrementaTiempo()
        
        filtros.filtrado(cap,fondo.getfondoHSV())
        
        circulos.contornoExterior(filtros.getFrameFiltrado(),filtros.getFrameOriginal())
        
        circulos.buscaCirculos(filtros.getFrameFiltrado(),filtros.getFrameOriginal())

        if (circulos.CirculoDetectado()):
            
            promedio.agregaPixel(circulos.getPromedio())
        
        elif (circulos.ObjetoDetectado()):
            #print 'N1'
            error = error +1
            if (error >= 5):
                #fondo.capturaFondoAgain(cap)
                error=0
                
            if ( 'N' is not UltimaLetra ):
                arduino.write('N')
                UltimaLetra = 'N'
            
        
        if (promedio.getPromedioReady()):
            #print 'ready'
            prom = promedio.getPixelPromedio()
            letra,color_detectado=color.ClasificadorColor(prom)
            if (letra is not UltimaLetra ):
                arduino.write(letra)
                error=0
                UltimaLetra = letra
            #time.sleep(0.05)
            #contador=contador+1
            #print contador
        enviar1.send([color_detectado,frame])
        while not q.empty():
            if((q.get())==4):
                cap.stop()
                exit()
                break

def inferfaz_grafica(recibir1,q):  
    class TapitasAPP(tk.Tk):
        def __init__(self):
            tk.Tk.__init__(self)
            self.title("TestCV Sofware")
            self.geometry('600x600')
            self._frame = None
            self.switch_frame(StartPage)
            self.flag_seteo=0
            
            self.mainloop()
            
        def switch_frame(self, frame_class):
            """Destroys current frame and replaces it with a new one."""
            new_frame = frame_class(self)
            if self._frame is not None:
                self._frame.destroy()
            
            self._frame = new_frame
            self._frame.pack()      

    class StartPage(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            
            def exitProgramm():
                #exit()
                res = tkMessageBox.askyesno('Exit','Desea realmente salir del programa?')
                if res is True:
                    if self.master._job is not None:
                        self.master.after_cancel(self.master._job)
                        self.master._job = None
                    q.put(4)
                    [color,vivo] = recibir1.recv()
                    sys.exit()
            
            start_label = tk.Label(self, text="Bienvenido !",height=3, width=40)
            
            def clicked_proceso():
                if self.master._job is not None:
                    self.master.after_cancel(self.master._job)
                    self.master._job = None
                if(master.flag_seteo==0):
                    master.switch_frame(PreviousLabel)
                if(master.flag_seteo==1):
                    master.switch_frame(ProccessLabel)
            
            page_2_button = tk.Button(self, text="Inicio", command=clicked_proceso,height=3, width=40)
            page_3_button = tk.Button(self, text="Salir del sistema", command= exitProgramm,height=3, width=40,bg = "red")
            
            start_label.pack(side="top", fill="x", pady=10)
            page_2_button.pack()
            page_3_button.pack()
            
            [color_obtenido,vivo] = recibir1.recv()
            self.updater(master)
        def updater(self,master):
            [color_obtenido,vivo] = recibir1.recv()        
            self.master._job =self.master.after(35, self.updater, master)
            
    class PreviousLabel(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            
            def clicked_menu():
                if self.master._job is not None:
                    self.master.after_cancel(self.master._job)
                    self.master._job = None
                master.switch_frame(StartPage)
            
            start_button = tk.Button(self, text="Regresar al menu principal", command=clicked_menu,height=3, width=40)
            start_button.pack(side = BOTTOM)
            
            def clicked_proceso():
                q.put(5)
                if self.master._job is not None:
                    self.master.after_cancel(self.master._job)
                    self.master._job = None
                master.flag_seteo=1
                master.switch_frame(ProccessLabel)
            
            capturarfondo_button = tk.Button(self, text="Capturar fondo y comenzar", command=clicked_proceso,height=3, width=40)
            capturarfondo_button.pack()
            
            #RECIBO EL FRAME POR PIPELINING
            [color_obtenido,vivo] = recibir1.recv()
            vivo = cv2.resize(vivo, (0,0), fx=0.5, fy=0.5) 
            vivo=cv2.cvtColor(vivo,cv2.COLOR_BGR2RGB)
            vivo=Image.fromarray(vivo)
            vivo=ImageTk.PhotoImage(vivo)             
            Imagenntomado = tk.Label(self, image = vivo)
            Imagenntomado.pack()
            Imagenntomado.image=vivo            
            self.updater(master, Imagenntomado)
        def updater(self,master, Imagenntomado):
            [color_obtenido,vivo] = recibir1.recv()
            vivo = cv2.resize(vivo, (0,0), fx=0.5, fy=0.5) 
            vivo=cv2.cvtColor(vivo,cv2.COLOR_BGR2RGB)
            vivo=Image.fromarray(vivo)
            vivo=ImageTk.PhotoImage(vivo)              
            Imagenntomado.configure(image=vivo)
            Imagenntomado.image = vivo            
            self.master._job =self.master.after(35, self.updater, master, Imagenntomado)    
            
    class ProccessLabel(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            proccess_label = tk.Label(self, text="Color detectado:",height=3, width=40)
            proccess_label.pack(side=TOP, fill="x", pady=10)
            
            def clicked_menu():
                if self.master._job is not None:
                    self.master.after_cancel(self.master._job)
                    self.master._job = None
                master.switch_frame(StartPage)
            
            start_button = tk.Button(self, text="Regresar al menu principal", command=clicked_menu,height=3, width=40)
            start_button.pack(side = BOTTOM)
           
            lbl = tk.Label(self, text=" ",height=3, width=40)
            lbl.pack(side=TOP)
            
            [color_obtenido,vivo] = recibir1.recv()
            self.updater(master,lbl)
        def updater(self,master,lbl):
            [color_obtenido,vivo] = recibir1.recv()
            lbl.configure(text= color_obtenido)
            self.master._job =self.master.after(5, self.updater, master,lbl)

    app=TapitasAPP()

if __name__ == "__main__":

    enviar1, recibir1 = mp.Pipe()
    q = mp.Queue()
    
    P_capturar = mp.Process(target=captura_frame, args=(enviar1,q))

    P_interfaz = mp.Process(target=inferfaz_grafica, args=(recibir1,q))
    
    P_capturar.start()
    P_interfaz.start()
