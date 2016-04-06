#Author: Lum Ramabaja
#--------------------

#Script takes as an input a blood smear image and returns a folder, 
#containing every red blood cell of the blood smear as a separate image.
#The first image in the directory "00.png" shows the original image, with the rectangles that were used to segment the cells.
  
#Input: blood smear image, directory name.
#Example: python Project_RBC_Segmentation.py Blood_Image.png Diractory_path 
#-----------------------------------------------------------------------------

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import sys
import os

#########################################################################

#Checking if input makes sense.
img = sys.argv[1]
directory = sys.argv[2]
if not os.path.isdir(directory):
    os.makedirs(directory)
if not os.path.exists(img):
    sys.exit("Image does not exist.")
#read image
image = cv2.imread(img)
img_copy = image.copy()

#Assign a value to num, depending on how many files are in the directory.
#We use num to know how to name the files.
if os.listdir(directory) == []: 
    num = 1
else: 
    #number of files in the directory
    num = len([f for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))])

#########################################################################


#convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#improve the contrast of our images
darker = cv2.equalizeHist(gray)
#Otsu's threshod
ret,thresh = cv2.threshold(darker,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#invert
newimg = cv2.bitwise_not(thresh)
#find contours of newimg
im2, contours, hierarchy = cv2.findContours(newimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#filling the "holes" of the cells
for cnt in contours:
    cv2.drawContours(newimg,[cnt],0,255,-1)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(newimg)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=newimg)
 
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=newimg)


# loop over the unique labels returned by the Watershed algorithm
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    if label == 0:
        continue
 
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
 
    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)

    #saving every single cell as a rectangular image.
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,0),2)
    roi = image[y:y+h, x:x+w]
    cv2.imwrite(directory + "/cell{}.png".format(num), roi)
    num = num + 1

#save an image where every cell is segmented by rectangles.
cv2.imwrite(directory + "/00.png", img_copy)
