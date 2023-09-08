import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

QUALITY_STEPS = [5, 25, 50, 75, 95]
IMAGE_PATH = 'example/reh.png'

# Load .png image
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
img_encode = cv2.imencode('.jpg', img)[1]
img_dec = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)

# create figure
fig = plt.figure("JPG Compression", figsize=(15, 5))
fig.tight_layout()
fig.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
  
# setting values to rows and column variables
rows = 1
columns = len(QUALITY_STEPS)+1

# plot
axes = fig.add_subplot(rows, columns, 1)
plt.imshow(img)
plt.title("ORIGINAL")
plt.xlabel(str(os.path.getsize("example/reh.png")) + " Bytes")

for i in range(len(QUALITY_STEPS)):
  # compress and encode as jpg
  img_encode = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY_STEPS[i]])[1]
  img_dec = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)

  fig.add_subplot(rows, columns, 2+i, sharex=axes, sharey=axes)

  # showing image
  plt.imshow(img_dec)
  plt.title("(" + str(i+1) + ") Q " + str(QUALITY_STEPS[i]))
  plt.xlabel(str(len(img_encode)) + " Bytes")
  
plt.show()