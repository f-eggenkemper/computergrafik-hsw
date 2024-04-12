import cv2
import matplotlib.pyplot as plt
import numpy as np
from pygments.formatters import img

path = "../Images/Thermobecher.jpg"
image_BGR = cv2.imread(path)
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

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

def create_image_blurred(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_blurred = cv2.GaussianBlur(image, (111, 111), 0)
    image_blurred_dictionary = {"title": "Blurred Image", "image": image_blurred}
    return image_blurred_dictionary


def grabcut_image(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (830, 1057, 1200, 2500)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return {"title": "Grabcut Maske", "image" : mask2}


def image_merge(image_original, image_blurred, mask):
    image_merged = image_blurred.copy()
    for x in range (image_original.shape[0]):
        for y in range (image_original.shape[1]):
            if mask[x][y] == 1:
                image_merged[x][y] = image_original[x][y]
    image_merged_dictionary = {"title": "Merged Image", "image" : image_merged}
    return image_merged_dictionary

def image_cut_before(image, mask):
    image_cut = image.copy()
    for x in range (image_cut.shape[0]):
        for y in range (image_cut.shape[1]):
            if mask[x][y] == 1:
                image_cut[x][y] = [158,158,158]
    image_cut_dictionary = {"title": "Cut Image", "image" : image_cut}
    return image_cut_dictionary

def create_dilated_mask(mask):
    dilated_mask = mask.copy()
    dilated_mask = cv2.dilate(dilated_mask, kernel=np.ones((5, 5), np.uint8), iterations=3)
    image_dilated_dictionary = {"title": "Dilated Image", "image": dilated_mask}
    return image_dilated_dictionary

def create_closed_mask(mask):
    closed_mask = mask.copy()
    closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8), iterations=3)
    image_closed_dictionary = {"title": "Closed Image", "image": closed_mask}
    return image_closed_dictionary

#def create_inpainted_image(image, mask):
    
    

image_masked = grabcut_image(image_RGB)
image_masked = image_masked["image"]
#image_dilated = create_dilated_mask(image_masked)
image_closed = create_closed_mask(image_masked)
#image_dilated = image_dilated["image"]
image_closed = image_closed["image"]
image_cut = image_cut_before(image_BGR, image_masked)
image_cut = image_cut["image"]
image_blurred = create_image_blurred(image_cut)
image_blurred = image_blurred["image"]


image_array = []
image_array.append(create_original_image(image_BGR))
#image_array.append(grabcut_image(image_RGB))
#image_array.append(create_dilated_mask(image_masked))
#image_array.append(image_merge(image_RGB,image_blurred,image_dilated))
image_array.append(image_merge(image_RGB,image_blurred,image_closed))
#image_array.append(image_cut_before(image_RGB, image_masked))

plot_images(image_array)
