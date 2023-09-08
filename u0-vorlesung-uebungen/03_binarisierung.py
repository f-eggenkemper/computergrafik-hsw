#Import the necessary packages
import cv2
import matplotlib.pyplot as plt

import cv2

# ---------- #
# READ IMAGE #
# ---------- #
path='../example/greenscreen.jpg' 
img=cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------- #
# Global Threshold #
# ---------------- #
_,img_global = None #TODO: Binarisierung mit globalem Threshold

# --------------- #
# Local Threshold #
# --------------- #
img_local = None #TODO: Binarisierung mit lokalem Threshold

# -------- #
# PLOTTING #
# -------- #
images = [img, img_global, img_local]
titles = ['Original Image','Global','Local']
for i in range(len(images)):
  plt.subplot(len(images)/3,3,i+1),plt.imshow(images[i],'gray')
  plt.title(titles[i]), plt.xticks([]), plt.yticks([])
plt.show()


