import numpy as np
import cv2
import json
from matplotlib import pyplot as plt

image_src = cv2.imread('../example/schriftzug.png', 0)
# WICHTIG: Morphologische Operationen in OpenCV immer mit Weiß als Vordergrund
image_src = cv2.bitwise_not(image_src)
kernels = []

kernel_cross_3 = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype="uint8")
kernels.append(['Raute (3)', kernel_cross_3])

kernel_cross_5 = np.array([[0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0],
                           [1, 1, 1, 1, 1],
                           [0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0]], dtype="uint8")
kernels.append(['Raute (5)', kernel_cross_5])

kernel_5 = np.array([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]], dtype="uint8")
kernels.append(['Rechteck (5)', kernel_5])

kernel_schraeg = np.array([[0, 0, 1, 1, 1],
                          [0, 1, 1, 1, 0],
                          [1, 1, 1, 0, 0]], dtype="uint8")
kernels.append(['Schräges El.', kernel_schraeg])

images_eroded = []
images_dilated = []
images_opening = []
images_closing = []
for i in range(len(kernels)):
    images_eroded.append(cv2.erode(image_src, kernels[i][1]))
    images_dilated.append(cv2.dilate(image_src, kernels[i][1]))
    images_opening.append(cv2.dilate(cv2.erode(image_src, kernels[i][1]), kernels[i][1]))
    images_closing.append(cv2.erode(cv2.dilate(image_src, kernels[i][1]), kernels[i][1]))

cols = ['Original', 'Dilation', 'Erosion', 'Opening', 'Closing']
rows = ['{}'.format(k[0]) for k in kernels]

fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(12, 8), sharex=True, sharey=True)

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, rotation=90, size='large')

for ax, row in zip(axes[:, 0], rows):
    ax.imshow(image_src, cmap='gray')

for i, ax in enumerate(axes[:, 1]):
    ax.imshow(images_dilated[i], cmap='gray')

for i, ax in enumerate(axes[:, 2]):
    ax.imshow(images_eroded[i], cmap='gray')

for i, ax in enumerate(axes[:, 3]):
    ax.imshow(images_opening[i], cmap='gray')

for i, ax in enumerate(axes[:, 4]):
    ax.imshow(images_closing[i], cmap='gray')


fig.tight_layout()
plt.show()
