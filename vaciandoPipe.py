from multiprocessing import Process, Pipe
import sys
import cv2
import time

def procesoPrincipal(enviar1):
    class VehiculoAutonomo (object):
        def __init__(self):
            self.frameCompleto = []
            self.cap = self._abrirCamara()

        def _abrirCamara (self):
            return cv2.VideoCapture(0)

        def comenzar(self):
            while self.cap.isOpened():
                ret, frameCompleto = self.cap.read()
                if ret:
                    self.frameCompleto = frameCompleto

                cv2.imshow('frameCompleto', frameCompleto)

                key = cv2.waitKey(10)
                if key == ord('s') or key == ord('S'):
                    enviar1.send('stop')
                if key == ord('d') or key == ord('D'):
                    enviar1.send('noise')
                if key == ord('q') or key == ord('Q'):
                    enviar1.send('exit')
                    break


            # When everything done, release the video capture object
            self.cap.release()
            # Closes all the frames
            cv2.destroyAllWindows()
            exit()

    vehiculoAutonomo = VehiculoAutonomo()
    vehiculoAutonomo.comenzar()

def procesoAuxiliar(recibir1):
    class controladorPWM (object):
        def __init__(self):
            self.hello = 0

        def start_loop(self):
            while True:
                orden = recibir1.recv()
                print(orden)
                if orden == 'exit':
                    sys.exit()
                elif orden == 'stop':
                    time.sleep(15)
                    while recibir1.poll():
                        print('cleaning: ', recibir1.recv())


    controladorPwm = controladorPWM()
    controladorPwm.start_loop()

if __name__ == "__main__":

    enviar1, recibir1 = Pipe()

    P_principal = Process(target=procesoPrincipal, args=(enviar1,))
    P_auxiliar = Process(target=procesoAuxiliar, args=(recibir1,))
    
    P_principal.start()
    P_auxiliar.start()
