import numpy as np
import cv2
from matplotlib import pyplot as plt

#####################################################################################################
# Dieses Beispiel zeigt die Erosion und Dilatation "manuell" mit numpy. 
# Auf die Anwendung verschiedener Kernel wurde bewusst verzichtet. 
# Es wird in diesem Beispiel stets mit einem Rechteck bzw. 8-Nachbarn-Kernel gearbeitet
# 
# Wichtiger Hinweis:
# - Der Grauwert ist umgedreht bei Python im Vergleich zur Literatur
# - Literatur: Der Grauwert 0 wird schwarz,  255 als weiß interpretiert.
# - Python: 255 schwarz, 0 weiß
# - dementsprechend sind max und min im vergleich zu den Folien hier "vertauscht"
#####################################################################################################


image_src = cv2.imread('../example/erosion_dilation.png', 0)

kernel = np.ones((3,3))

k_height, k_width = kernel.shape

flat_submatrices = np.array([
    image_src[i:(i + k_width), j:(j + k_height)]
    for i in range(image_src.shape[0]) for j in range(image_src.shape[1])
])

image_erode = np.array([i.max() for i in flat_submatrices])
image_erode = image_erode.reshape(image_src.shape)

image_dilated = np.array([i.min() for i in flat_submatrices])
image_dilated = image_dilated.reshape(image_src.shape)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(5, 5))
ax1.title.set_text('Original')
ax1.imshow(image_src, cmap='gray')
ax2.title.set_text("Eroded - {}".format(k_height))
ax2.imshow(image_erode, cmap='gray')
ax3.title.set_text("Dilated - {}".format(k_height))
ax3.imshow(image_dilated, cmap='gray')
plt.show()
