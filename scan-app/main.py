import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

imagesArray = []
titlesArray = []


def plot_images(images, titles):
    fig, axes = plt.subplots(len(images), len(images[0]), sharex=True, sharey=True)

    for i, axes2 in enumerate(axes):
        axes2[0].set_ylabel(titles[i], rotation=90)

        for j, key in enumerate(images[0]):
            axes[0][j].set_title(key)

        for img, ax in zip(images[i].values(), axes2):
            #ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.imshow(img, cmap=cm.gray)

    plt.show()


def gen_images(path):
    for img_file in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_file))
        img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        imagesArray.append(apply_effects(img))
        titlesArray.append(os.fsdecode(img_file))


def grayscale_image(img):
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_grayscale


def blur_image(img):
    img_blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
    return img_blurred


def do_closing_operation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(11, 5))
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel, iterations=5)
    return img_closed


def binarize_image(img):
    img_binarized = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 2)
    return img_binarized


def apply_canny_edge_detection(img):
    img_edges = cv2.Canny(img, 10, 20, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
    img_edges = cv2.morphologyEx(img_edges, cv2.MORPH_DILATE, kernel=kernel)
    return img_edges


def add_bounding_box(img, img_original):
    x, y, w, h = cv2.boundingRect(img)
    img_bounding_box = cv2.rectangle(img_original.copy(), (x, y), (x + w, y + h), (255, 0, 0), 10)
    return img_bounding_box


def draw_contours(img, img_original):
    img_contours = img_original.copy()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    epsilon = 0.1*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    for i in approx:
        cv2.circle(img_contours, i[0], 15, (255,0,0), -1)
    return img_contours, approx

def draw_hough_lines(img_binarized, img_original):
    linesArray = cv2.HoughLinesP(img_binarized, 1, np.pi/180, 150, None, 350, 10)
    img = img_original.copy()
    for i in range(0, len(linesArray)):
        l = linesArray[i][0]
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255), cv2.LINE_AA)
    return img

def perspective_transformation(img, points):
    img_transformed = img.copy()
    pts1 = np.float32([[points[0][0],points[1][0],points[2][0],points[3][0]]])
    width = img.shape[1]
    height = img.shape[0]
    pts2 = np.float32([[0,0], [0,height], [width,height], [width, 0]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img_transformed,M,(width,height))
    return dst

def apply_effects(img_original):
    img_grayscale = grayscale_image(img_original)
    img_blurred = blur_image(img_grayscale)
    img_closed = do_closing_operation(img_blurred)
    img_binarized = binarize_image(img_closed)
    img_lines = draw_hough_lines(img_binarized, np.zeros(img_grayscale.shape,dtype=np.uint8))
    img_contours, points = draw_contours(img_lines, np.zeros(img_grayscale.shape,dtype=np.uint8))
    img_perspective = perspective_transformation(img_original, points)

    return {
        "original": img_original,
        "grayscale": img_grayscale,
        "blurred": img_blurred,
        "closed": img_closed,
        "binarized": img_binarized,
        "lines": img_lines,
        "contours": img_contours,
        "result" : img_perspective
    }


if __name__ == "__main__":
    gen_images("./examples/")
    plot_images(imagesArray, titlesArray)
