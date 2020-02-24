import cv2
from pyzbar import pyzbar

def leer_qr(frame):

    barcodes = pyzbar.decode(frame)

    barcodeData = barcodes[0].data.decode("utf-8")

    return barcodeData