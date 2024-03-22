import cv2
import matplotlib.pyplot as plt
import numpy as np

path = "../Images/Thermobecher.jpg"
image_BGR = cv2.imread(path)
#image_RGB = cv2.cvtColor(image_BGR,cv2.COLOR_BGR2RGB) Simon sagt, er möchte es noch behalten.

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

    fig, ax = plt.subplots(1, len(img_arr), sharex=True, sharey=True)
    
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
    image_canny = cv2.Canny(image_binary, 100, 200)
    image_binary_dictionary = {"title": "Binary Image", "image": image_canny}
    return image_binary_dictionary


def create_blurred_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_blurred = cv2.GaussianBlur(image, (55, 55), 0)
    image_blurred_dictionary = {"title": "Blurred Image", "image": image_blurred}
    return image_blurred_dictionary


blurred_image = create_blurred_image(image_BGR)
blurred_image = blurred_image["image"]
blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)

image_array = []
image_array.append(create_original_image(image_BGR))
image_array.append(create_binary_image(blurred_image))
image_array.append(create_blurred_image(image_BGR))

plot_images(image_array)
