import cv2
from pyzbar import pyzbar

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )

i=1

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        barcodes = pyzbar.decode(frame)

        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # print("[INFO] found {} barcode {}".format(barcodeType, barcodeData))


        cv2.imshow('imagen', frame)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            break
        if key == ord('s') or key == ord('S'):
            continue
        if key == ord('c') or key == ord('C'):
            cv2.imwrite("lecturaQR"+str(i)+".jpg", frame)
            i += 1

cap.release()
# Closes all the frames
cv2.destroyAllWindows()