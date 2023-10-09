import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Button, Slider


imagesArray = []
titlesArray = []

def plot(images, titles):
    fig, axes = plt.subplots(len(images) + 1, len(images[0]), sharex=False, sharey=False, gridspec_kw={'height_ratios': [3, 1]})
    fig.tight_layout()

    # Color Mode
    axes[len(axes)-1][0].set_title('Color Mode')
    axes[len(axes)-1][0].axis('off')
    radio = RadioButtons(axes[len(axes)-1][0], ('Color', 'Grayscale', 'Binary'))
    radio.on_clicked(lambda label: change_color_mode(label, axes, fig))

    # Helligkeit
    helligkeit_slider = Slider(
        ax=axes[len(axes) - 1][1],
        label='Kontrast',
        valmin=0.1,
        valmax=3.0,
        valinit=1.0,
        valstep=0.1
        )
    helligkeit_slider.on_changed(lambda val: change_contrast(val, axes, fig))
    
    # Kontrast
    kontrast_slider = Slider(
        ax=axes[len(axes) - 1][2],
        label='Helligkeit',
        valmin=1,
        valmax=100,
        valinit=50,
        valstep=1
        )
    kontrast_slider.on_changed(lambda val: change_brightness(val, axes, fig))
    
    plot_images(images, titles, axes)

    plt.show()

def plot_images(images, titles, axes):
    for i, axes2 in enumerate(axes[0:len(axes)-1]):
        axes2[0].set_ylabel(titles[i], rotation=90)

        for j, key in enumerate(images[0]):
            axes[0][j].set_title(key)

        for img, ax in zip(images[i].values(), axes2):
            #ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.clear()
            ax.imshow(img, cmap=cm.gray)


def change_color_mode(label, axes, fig):
    if (label == 'Color'):
        for img in imagesArray:
            img['result'] = img['cropped']
    elif (label == 'Grayscale'):
        for img in imagesArray:
            img['result'] = grayscale_image(img['cropped'])
    elif (label == 'Binary'):
        for img in imagesArray:
            img['result'] = cv2.adaptiveThreshold(grayscale_image(img['cropped']), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)
    plot_images(imagesArray, titlesArray, axes)

    fig.canvas.draw_idle()

def change_brightness(val, axes, fig):
    for img in imagesArray:
        img['result'] = cv2.convertScaleAbs(img['cropped'], beta=val)
    plot_images(imagesArray, titlesArray, axes)
    fig.canvas.draw_idle()

def change_contrast(val, axes, fig):
    for img in imagesArray:
        img['result'] = cv2.convertScaleAbs(img['cropped'], alpha=val)
    plot_images(imagesArray, titlesArray, axes)
    fig.canvas.draw_idle()


def gen_images(path, idx = -1):
    if (idx != -1):
        list = [os.listdir(path)[idx]]

    for img_file in list:
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
    img_binarized = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 2)
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

def reduce_points(points, shape, corner_threshold = 200):
    width = shape[1]
    height = shape[0]
    ret_points = np.empty(4, dtype=object)
    ret_points[0] = next((point for point in points if (point[0][0] <= corner_threshold and point[0][1] <= corner_threshold)),[[0, 0]])
    ret_points[1] = next((point for point in points if (point[0][0] <= corner_threshold and point[0][1] >= (height - corner_threshold))),[[0, height]])
    ret_points[2] = next((point for point in points if (point[0][0] >= (width - corner_threshold) and point[0][1] >= (height - corner_threshold))), [[width, height]])
    ret_points[3] = next((point for point in points if (point[0][0] >= (width - corner_threshold) and point[0][1] <= corner_threshold)), [[width, 0]])
    return ret_points

def draw_contours(img, img_original):
    img_contours = img_original.copy()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    epsilon = 0.05*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    approx = reduce_points(approx, img_original.shape)
    for i in approx:
        cv2.circle(img_contours, i[0], 15, (255,0,0), -1)
    return img_contours, approx

def draw_hough_lines(img_binarized, img_original):
    linesArray = cv2.HoughLinesP(img_binarized, 1, np.pi/180, 150, None, 350, 10)
    img = img_original.copy()
    for i in range(0, len(linesArray)):
        l = linesArray[i][0]
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255), lineType=cv2.LINE_4)
    return img

def perspective_transformation(img, points):
    img_transformed = img.copy()
    width = img.shape[1]
    height = img.shape[0]
    pts1 = np.float32([[points[0][0],points[1][0],points[2][0],points[3][0]]])
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
        #"grayscale": img_grayscale,
        #"blurred": img_blurred,
        #"closed": img_closed,
        #"binarized": img_binarized,
        #"lines": img_lines,
        #"contours": img_contours,
        "cropped" : img_perspective,
        "result" : img_perspective
    }


if __name__ == "__main__":
    gen_images("./examples/", idx = 1)
    plot(imagesArray, titlesArray)
