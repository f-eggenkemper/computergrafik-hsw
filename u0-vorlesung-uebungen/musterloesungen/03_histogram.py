import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------- #
# READ IMAGE #
# ---------- #
path='../example/greenscreen.jpg' 
img=cv2.imread(path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------- #
# BUILD HISTOGRAM #
# --------------- #
histogram = np.zeros(256)

height=len(img[0])
width=len(img)

for i in range(width-1):
  for j in range(height-1):
    histogram[ img[i][j] ] += 1

# -------- #
# PLOTTING #
# -------- #
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(np.arange(0, 256, 1),histogram)
plt.show()

