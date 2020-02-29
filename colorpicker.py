import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.patches as mpatches
def color_picker():
            frame = cv2.imread('Imagenes/Imagen4.png')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
            plt.show()
            pixel_colors = frame.reshape((np.shape(frame)[0]*np.shape(frame)[1], 3))
            norm = colors.Normalize(vmin=-1.,vmax=1.)
            norm.autoscale(pixel_colors)
            pixel_colors = norm(pixel_colors).tolist()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1, projection="3d")
            axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
            axis.set_xlabel("Hue")
            axis.set_ylabel("Saturation")
            axis.set_zlabel("Value")
            plt.show()
if __name__ == "__main__":
    color_picker()