import cv2
import matplotlib.pyplot as plt
import numpy as np
from pygments.formatters import img

path = "../Images/Thermobecher.jpg"
image_BGR = cv2.imread(path)
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB) #Simon sagt, er möchte es noch behalten.

# Methode um Bilder darzustellen


def plot_single_image(image_dict):
    #Dictionary{"title":title, "image":img}
    plt.imshow(image_dict["image"], cmap="gray")
    plt.title(image_dict["title"])
    plt.show()


def plot_images(img_arr):
    # img_arr = [original, binär, blurr, kanten]
    # original = Dictionary{"title":title, "image":img}
    if len(img_arr) == 0:
        print("No images")
        return

    if len(img_arr) == 1:
        plot_single_image(img_arr[0])
        return

    fig, ax = plt.subplots(1, len(img_arr), sharex=False, sharey=False)
    
    for i in range(0, len(img_arr)):
        ax[i].axis("off")
        ax[i].imshow(img_arr[i]["image"], cmap="gray")
        ax[i].set_title(img_arr[i]["title"])

    plt.show()


def create_original_image(image):

    original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image_dictionary = {"title": "Original Image", "image": original_image_rgb}

    return original_image_dictionary


def create_binary_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_blurred = cv2.GaussianBlur(image, (55, 55), 0)
    image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2GRAY)
    #_, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    image_binary = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0.5)
    #image_canny = cv2.Canny(image_binary, 100, 200)
    image_binary_dictionary = {"title": "Binary Image", "image": image_binary}
    return image_binary_dictionary

def create_canny_image(image):
        image_canny = cv2.Canny(image, 100, 200.0)
        image_canny_dictionary = {"title": "Canny Image", "image": image_canny}
        return image_canny_dictionary



def create_image_blurred(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_blurred = cv2.GaussianBlur(image, (55, 55), 0)
    image_blurred_dictionary = {"title": "Blurred Image", "image": image_blurred}
    return image_blurred_dictionary

def create_cutted_image(image):
    x1, y1 = 830, 1057
    x2, y2 = 2069, 3619
    image_cutted = image[y1:y2, x1:x2]
    return image_cutted

def grabcut_image(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (830, 1057, 1200, 2500)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    plt.imshow(img), plt.colorbar(), plt.show()


grabcut_image(image_RGB)
# Bild zuschneiden auf den relevanten Bereich
image_cutted = create_cutted_image(image_BGR)


image_blurred = create_image_blurred(image_cutted)
image_blurred = image_blurred["image"]
image_blurred = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2BGR)
image_binary = create_binary_image(image_blurred)
image_binary = image_binary["image"]






image_array = []
image_array.append(create_original_image(image_BGR))
image_array.append(create_binary_image(image_blurred))
image_array.append(create_image_blurred(image_BGR))
image_array.append(create_canny_image(image_binary))

plot_images(image_array)
