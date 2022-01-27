from multiprocessing import Process, Pipe
import cv2
import sys

def procesoPrincipal(enviar1):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
        
            cv2.imshow('imagen', frame)
            key = cv2.waitKey(10)
            if key == ord('s') or key == ord('S'):
                enviar1.send('1')
            if key == ord('q') or key == ord('Q'):
                # sys.exit()
                enviar1.send('stop')
                sys.exit()
                break

    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def procesoAuxiliar(recibir1):
    orden = '0'
    while True:
        orden = recibir1.recv()
        print(orden)
        if orden == 'stop':
            sys.exit()


if __name__ == "__main__":

    enviar1, recibir1 = Pipe()
    
    P_principal = Process(target=procesoPrincipal, args=(enviar1,))
    P_auxiliar = Process(target=procesoAuxiliar, args=(recibir1,))
    
    P_principal.start()
    P_auxiliar.start()