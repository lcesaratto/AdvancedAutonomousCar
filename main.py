from deteccionLineas import *
from deteccionObjetos import *
from detectorQR import *

if __name__ == "__main__":
    leer_qr(frame)

    seguidorLineas = SeguimientoLineas()
    seguidorLineas.run()
    
    buscadorObjetos = DetectorObjetos()
    class_ids = buscadorObjetos.buscarObjetos(frame)