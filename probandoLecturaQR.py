import cv2
from pyzbar import pyzbar

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        barcodes = pyzbar.decode(frame)
        if barcodes:
            qr_encontrado = barcodes[0].data.decode("utf-8")
            print(qr_encontrado[0])

        cv2.imshow('imagen', frame)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            break

cap.release()
# Closes all the frames
cv2.destroyAllWindows()