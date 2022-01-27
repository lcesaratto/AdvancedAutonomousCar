import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.patches as mpatches

import copy 
def color_picker():
            frame = cv2.imread('LineaRoja.png')
            frameRojo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
            plt.show()
            pixel_colors = frameRojo.reshape((np.shape(frameRojo)[0]*np.shape(frameRojo)[1], 3))
            norm = colors.Normalize(vmin=-1.,vmax=1.)
            norm.autoscale(pixel_colors)
            pixel_colors = norm(pixel_colors).tolist()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS) #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1, projection="3d")
            axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
            axis.set_xlabel("Hue")
            axis.set_ylabel("Luminescese") #Saturation
            axis.set_zlabel("Saturation") #Value
            plt.show()
            




def _detectarLineaVerde(frame):
            #Corto el frame
            # frame = self.frameProcesado
            # frame = frameOriginal[320:480,0:640] #
            # cv2.imshow('FrameOriginalRecortado', frame)
            #Defino parametros HSV para detectar color verde 
            lower_green = np.array([50, 80, 20])
            upper_green = np.array([75, 120, 60])
            
            #Aplico filtro de color con los parametros ya definidos
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            hls_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            mask_green = cv2.inRange(hls_green, lower_green, upper_green)
            # self.mask_green = copy.deepcopy(mask_green)
            # kernel = np.ones((3,3), np.uint8)
            # mask_green_a = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
            # kernel = np.ones((5,5), np.uint8)
            # mask_green_e = cv2.dilate(mask_green, kernel, iterations=1)
            #kernel = np.ones((11,11), np.uint8)
            #mask_green_c = cv2.morphologyEx(mask_green_e, cv2.MORPH_CLOSE, kernel)
            cv2.imshow('FiltroVerde', mask_green)
        


if __name__ == "__main__":
    color_picker()
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #         ret, frameCompleto = cap.read()
    #         if ret:
    #             _detectarLineaVerde(frameCompleto)

    #         key = cv2.waitKey(10)
    #         if key == ord('q') or key == ord('Q'):
    #             break

    # cap.release()
    # # Closes all the frames
    # cv2.destroyAllWindows()