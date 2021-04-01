import matplotlib.pyplot as plt 
import numpy as np; import cv2; import os; import math
from PIL import Image, ImageFilter
#directories
img_import, img_export = "main_directory/", "processed_images/"

def cropping(img):
    # Crops the image to the largest possible square
    h, w = img.shape[:2]; min_h_w = min(h,w) # Find constraining dimension
    diff = [abs(min_h_w-h), abs(min_h_w-w)]
    # new contours
    min_w, max_w = int(diff[1]/2), int(w-diff[1]/2)
    min_h, max_h = int(diff[0]/2), int(h-diff[0]/2)
    return img[min_w:max_w,min_h:max_h]

def matrix(img): 
    # convert pictures to matrix with entries between 0 and 1
    A = cv2.imread(img_import + img); return np.array(A) / 255

##Function to change contrast
def change_contrast(img, contrast):
    # @param: img, path of image to change the contrast
    # @parem: contrast, value form 1 to 3
    #image = cv2.imread(img_import + img) 
    new_image = np.array(img)
    image = img
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(contrast*image[y,x,c], 0, 255)
    return new_image

def find_edges(img):
    #img: image name
    Filter = (-1, -1, -1, -1, 8, -1, -1, -1, -1)
    img = Image.open(img); 
    img = img.convert("L")
    # Calculating Edges with Laplace Kernel
    final = img.filter(ImageFilter.Kernel((3, 3), Filter, 1, 0))
    rgb_im = final.convert('RGB')
    matrix = np.array(rgb_im)
    #r, g, b = rgb_im.getpixel((1, 1))
    return matrix

#Import pictures to gray white and blur
kernelDimensions = (1,1); i= 0
dimensionsIMG = (320, 180) # change dimensions - still need to crop
for image in os.listdir(img_import):
    if i >= 18 and i <= 21:
        print("Working on image ", i)
        img_dir = img_import + "/" + image
        img = cv2.imread(img_dir)
        img = cv2.resize(img, dimensionsIMG)
        cv2.imwrite(img_import + "resized/" + image, img)
        #height,width = img.shape[:2]
        #image = cropping(image)
        image_edges = find_edges(img_import + "resized/" + image) # resize image -

        image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        image_rgb = cv2.blur(img, kernelDimensions) 
        image_rgb = change_contrast(image_rgb, 1.2)

        # SAVING IMAGES
        cv2.imwrite(img_export + "bw/" + image, image_bw)
        cv2.imwrite(img_export + "rgb/" + image, image_rgb)
        cv2.imwrite(img_export + "edges/" + image, image_edges)
    i += 1

print("DONE")
#Image to normalized Matrix 0-255:
#https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html