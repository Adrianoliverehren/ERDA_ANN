import math
import matplotlib.pyplot as plt 
import numpy as np
#import tensorflow as tf
#from keras.datasets import mnist
import cv2
import os
from PIL import Image, ImageFilter

img_import = "main_directory/"
img_export = "processed_images/"

def cropping(img):
    # Crops the image to the largest possible square
    h, w = img.shape[:2]
    min_h_w = min(h,w)
    diff = [abs(min_h_w-h), abs(min_h_w-w)]
    
    # new contours
    min_w, max_w = int(diff[1]/2), int(w-diff[1]/2)
    min_h, max_h = int(diff[0]/2), int(h-diff[0]/2)
    image=img[min_w:max_w,min_h:max_h]
    return image


def matrix(img):
    A = cv2.imread(img_import + img)
    A = np.array(A) / 255
    
    return A
print(matrix("blackandwhite.jpg"))


##Function to change contrast
def change_contrast(img, contrast):
    #img: path of image to change the contrast
    #contrast: value form 1 to 3

    image = cv2.imread(img_import + img)
    new_image = np.array(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(contrast*image[y,x,c], 0, 255)

    return new_image

print("NEXT")

#print(change_contrast("blackandwhite.jpg", 1.2))



# do for for filename in os.listdir():
"""
def detect_edges(img):
    #img: name of image to detect edges
    Filter = (-1, -1, -1, -1, 8, -1, -1, -1, -1)
    img = Image.open(img_import + img)
    img = img.convert("L")

    # Calculating Edges with Laplace Kernel
    final = img.filter(ImageFilter.Kernel((3, 3), Filter, 1, 0))

    final.save("EDGE_{}.jpg".format(img))

    #rgb_im = final.convert('RGB')
    #r, g, b = rgb_im.getpixel((1, 1))

    #Save the image
    #final.save("EDGE_{}.jpg".format(img_name))

    return [r,g,b]
"""

img_name = "a320neo_easyjet1.jpg"

def find_edges(img):
    #img: image name
    Filter = (-1, -1, -1, -1, 8, -1, -1, -1, -1)
    img = Image.open(img_import + img_name)
    img = img.convert("L")

    # Calculating Edges with Laplace Kernel
    final = img.filter(ImageFilter.Kernel((3, 3), Filter, 1, 0))

    rgb_im = final.convert('RGB')
    matrix = np.array(rgb_im)
    #r, g, b = rgb_im.getpixel((1, 1))

    return matrix

print(find_edges("a320neo_easyjet1.jpg"))
#find_edges("a320neo_easyjet1.jpg")
#cv2.imshow("Image", detect_edges("a320neo_easyjet1.jpg"))
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#Import pictures to gray white and blur
kernelDimensions = (3,3)
dimensionsIMG = (255, 255) # change dimensions - still need to crop
i= 0
for image in os.listdir(img_import):
    if i == 15:
        directory = img_import + "/" + image
        image = cv2.imread(directory)
        height,width=image.shape[:2]
        #image = cropping(image)
        image = cv2.resize(image, dimensionsIMG) # resize image - 
        image = cv2.blur(image,kernelDimensions)
        Testgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Image", Testgray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    i += 1
#Image to normalized Matrix 0-255:


"""
Code to edit contrast
normal_image = cv2.imread(img_import + "a320neo_easyjet1.jpg")
cv2.imshow("original img", normal_image)
cv2.imshow("Hella contrast", change_contrast(img_import + "a320neo_easyjet1.jpg", 1.3))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#change_contrast(img_import + "737-900_Delta_1.jpg", 0)

#https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html


