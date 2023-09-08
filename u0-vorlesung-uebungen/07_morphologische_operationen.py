import numpy as np
import cv2
import json
from matplotlib import pyplot as plt

# ---------- #
# READ IMAGE #
# ---------- #
image_src = cv2.imread('../example/erosion_dilation.png', 0)

# ------ #
# KERNEL #
# ------ #
kernels = []
# TODO : HIER BITTE DIE KERNEL ERGAENZEN

# -------------------------- #
# MORPHOLOGISCHE OPERATIONEN #
# -------------------------- #
# TODO: Hier Opening und Closing ergänzen (und Plotten nicht vergessen)
images_eroded = []
images_dilated = []
for i in range(len(kernels)):
    images_eroded.append(cv2.erode(image_src, kernels[i][1]))
    images_dilated.append(cv2.dilate(image_src, kernels[i][1]))

# -------- #
# PLOTTING #
# -------- #
cols = ['Original', 'Dilation', 'Erosion']
rows = ['{}'.format(k[0]) for k in kernels]

fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(12, 8))

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, rotation=0, size='large')

for ax, row in zip(axes[:, 0], rows):
    ax.imshow(image_src, cmap='gray')

for i, ax in enumerate(axes[:, 1]):
    ax.imshow(images_dilated[i], cmap='gray')

for i, ax in enumerate(axes[:, 2]):
    ax.imshow(images_eroded[i], cmap='gray')

fig.tight_layout()
plt.show()
