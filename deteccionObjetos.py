import cv2
import numpy as np
import time

# Load Yolo
class DetectorObjetos (object):
    def __init__(self):
        self.net = self._cargarModelo()
        self.classes = self._cargarClases()

        self.height = 480
        self.width = 640
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.font = cv2.FONT_HERSHEY_PLAIN

    def _cargarModelo(self):
        return cv2.dnn.readNet("4class_yolov3-tiny_final.weights", "4class_yolov3-tiny.cfg")
    
    def _cargarClases(self):
        classes = []
        with open("4classes.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def buscarObjetos (self, frame, mostrarResultado=False, retornarBoxes=False, retornarConfidence=False, calcularFPS=False):
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
        for i in range(len(boxes)):
                if not i in indexes:
                    del boxes[i]
                    del confidences[i]
                    del class_ids[i]

        if calcularFPS:
            print(1/(time.time()-tiempo_inicial))

        if mostrarResultado:
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label == 'SemaforoRojo' or label== 'SemaforoDos':
                    color = self.colors[0] #indice va de 0 a 3 para 4 clases
                elif label == 'SemaforoVerde':
                    color = self.colors[1] #indice va de 0 a 3 para 4 clases
                elif label == 'CartelUno':
                    color = self.colors[2] #indice va de 0 a 3 para 4 clases
                elif label == 'CartelCero':
                    color = self.colors[3] #indice va de 0 a 3 para 4 clases
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), self.font, 3, color, 2)

            cv2.imshow('window-name',frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        if not retornarBoxes and not retornarConfidence:
            return class_ids
        if retornarBoxes and retornarConfidence:
            return class_ids, boxes, confidences
        elif retornarBoxes:
            return class_ids, boxes
        else:
            return class_ids, confidences
