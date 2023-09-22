import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_images(images, titles):
    fig, axes = plt.subplots(1, len(images), sharex=True, sharey=True)

    for i, (ax, title) in enumerate(zip(axes.ravel(), titles)):
        ax.imshow(images[i], cmap=cm.gray)
        ax.set_title(title)  # Add the title
        ax.axis('off')

    plt.show()

imagesArray = []
titlesArray = []

img1 = cv2.imread("./examples/img1.jpeg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
imagesArray.append(img1)
titlesArray.append("Grayscale")

img_blurred = cv2.GaussianBlur(img1, ksize=(35,35), sigmaX=0)
imagesArray.append(img_blurred)
titlesArray.append("Blurred")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,ksize=(5,5))
dilated = cv2.morphologyEx(img_blurred, cv2.MORPH_CLOSE, kernel=kernel, iterations=15)
imagesArray.append(dilated)
titlesArray.append("Closed")

img_binarized = cv2.adaptiveThreshold(dilated,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,20)
imagesArray.append(img_binarized)
titlesArray.append("Binarized")

img_eroded = cv2.morphologyEx(img_binarized, cv2.MORPH_ERODE, kernel, iterations=5)
imagesArray.append(img_eroded)
titlesArray.append("Eroded")



edges = cv2.Canny(img_eroded, 100, 200)
imagesArray.append(edges)
titlesArray.append("Edges")

x,y,w,h = cv2.boundingRect(img_binarized)
print(x,y,w,h)
boundingbox_img = cv2.rectangle(img1, (x,y), (x+w,y+h), (255,0,0), 10)
imagesArray.append(boundingbox_img)
titlesArray.append("Box")

#eroded = cv2.erode(edges, kernel=kernel)


plot_images(imagesArray, titlesArray)
