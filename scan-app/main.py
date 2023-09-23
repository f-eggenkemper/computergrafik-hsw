import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

imagesArray = []
titlesArray = []
stepsArray = []

def plot_images(images, titles, steps):
    fig, axes = plt.subplots(len(images[0]), len(images), sharex=True, sharey=True)
    for i, (axes2, title, step) in enumerate(zip(axes, titles, steps)):
        axes2[0].set_ylabel(title, rotation=90)  # Add the title
        axes[0][i].set_title(step)
        for img,ax in zip(images[i],axes2):
            ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.imshow(img, cmap=cm.gray)

    plt.show()

def gen_images(path):
    stepsArray.append("Greyscale")
    for imgfile in os.listdir(path):
        img = cv2.imread(os.path.join(path, imgfile))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imagesArray.append(applyEffects(img))
        titlesArray.append(os.fsdecode(imgfile))

def blur(img):
    img_blurred = cv2.GaussianBlur(img, ksize=(35,35), sigmaX=0)
    stepsArray.append("Blurred")
    return img_blurred

def close(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,ksize=(5,5))
    dilated = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel, iterations=15)
    stepsArray.append("Closed")
    return dilated

def binarize(img):
    img_binarized = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,20)
    stepsArray.append("Binarized")
    return img_binarized

def genBoundingBox(img, original): 
    x,y,w,h = cv2.boundingRect(img)
    boundingbox_img = cv2.rectangle(original.copy(), (x,y), (x+w,y+h), (255,0,0), 10)
    stepsArray.append("Box")
    return boundingbox_img

def applyEffects(img):
    blurred = blur(img)
    closed = close(blurred)
    binarized = binarize(closed)
    boundingBox = genBoundingBox(binarized, img)
    return[img,blurred,closed,binarized,boundingBox]

gen_images("./examples/")


#img_eroded = cv2.morphologyEx(img_binarized, cv2.MORPH_ERODE, kernel, iterations=5)
#imagesArray.append(img_eroded)
#titlesArray.append("Eroded")

#edges = cv2.Canny(img_eroded, 100, 200)
#imagesArray.append(edges)
#titlesArray.append("Edges")

#eroded = cv2.erode(edges, kernel=kernel)


plot_images(imagesArray, titlesArray, stepsArray)
