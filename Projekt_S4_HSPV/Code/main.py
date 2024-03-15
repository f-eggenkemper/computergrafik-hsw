import cv2
import matplotlib.pyplot as plt
import numpy as np

path = "../Images/Thermobecher.jpg"
image_BGR = cv2.imread(path)
image_RGB = cv2.cvtColor(image_BGR,cv2.COLOR_BGR2RGB)

# Methode um Bilder darzustellen
def plot_image(img_arr):
    # img_arr = [original, binär, blurr, kanten]
    # original = Dictionary{"title":title, "image":img}
    
    fig, ax = plt.subplots(1, len(img_arr))
    
    for i in img_arr:
        
        

    
