from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2


# ---------- #
# READ IMAGE #
# ---------- #
path='../example/ente2.jpg' 
img_original =cv2.imread(path)
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_original = cv2.resize(img_original, (512, 512))

# ----------------- #
# FOURIER TRANSFORM #
# ----------------- #
img_fourier = fftshift(fft2(img_original))

# ---- #
# MASK #
# ---- #
img_masked = img_fourier.copy()
img_masked[252:260, 0:252] = 1
img_masked[252:260, 260:512] = 1

# -------------- #
# INVERT FOURIER #
# -------------- #
img_final = ifft2(ifftshift(img_masked)).real

# -------- #
# PLOTTING #
# -------- #
_, ((plt1, plt2),(plt3, plt4)) = plt.subplots(2, 2, sharex='col', sharey='row')
plt1.imshow(img_original, cmap=cm.gray)
plt2.imshow(np.log(abs(img_fourier)), cmap=cm.gray)
plt3.imshow(np.log(abs(img_masked)), cmap=cm.gray)
plt4.imshow(img_final, cmap=cm.gray)

plt.show()