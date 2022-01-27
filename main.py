from lib.deteccionLineas import procesoPrincipal
from lib.controlPWM import procesoAuxiliar
from multiprocessing import Process, Pipe

if __name__ == "__main__":
    enviar1, recibir1 = Pipe()
    
    P_principal = Process(target=procesoPrincipal, args=(enviar1,))
    P_auxiliar = Process(target=procesoAuxiliar, args=(recibir1,))
    
    P_principal.start()
    P_auxiliar.start()