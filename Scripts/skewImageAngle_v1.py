# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:05:00 2021

@author: sidne
"""

###########################################
# Sidney CB
# Created: 1/1/2021 (happy new year!)
# DescriptionFirst version of program that will read in a pdf document and 
# skew/tilt it based on a specified angle

# last edited:
# Changes:
###########################################

# import necessary packages
from pdf2image import convert_from_path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils



def findRectangle(image):
    # function to find a rectangle in the image
    
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our cutting surface in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)
        
    # find the area of the rectangle and the bounding points
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    return box
    
    
    
    


def perspectiveTransform1(img):
    # See https://stackoverflow.com/questions/33497736/opencv-adjusting-photo-with-skew-angle-tilt

    rows, cols, ch = img.shape  
    
    # Define the four points needed for a successful perspective transform
    # The first set identifies the four points of interest that are the 
    #   four corners of the object we are transforming
    # The second set of points identifies where each of the first points 
    #   will be transferred TO (in the output image)
    # This makes more sense how you can both skew and un-skew an image
    
    # FUTURE VERSION: auto detect the 1x1 inch calibration cube (or mat?) and  
    #    use those points to define the transformation matrix points
    
    # NOTE: probably will be some scaling issues... Since Python's screen is only so big. 
    #   May need to save as pdf with correct scaling (rather than using only python)
    
    print("Col size: "+str(cols));
    print("Row size: "+str(rows));
    
    originPts = np.float32(
        [[cols*.10, rows*.95],
         [cols*.90, rows*.95],
         [cols*.10, 0],
         [cols*.90, 0]]
    )
    
    newCols = 800; newRows = 600;
    targetPts = np.float32(
        [[0, 0],
         [newCols, 0],
         [0, newRows],
         [newCols, newRows]]
    )    
    
    # get the perspective transform
    M = cv2.getPerspectiveTransform(originPts, targetPts)
    # warp the image using the transform matrix
    # here the output size (dsize) is the same as the input size. Should change
    #   since the current image is so big. Likely change to be similar to 
    #   projector size?
    # May be able to use a 'flag' keyword to create the inverse transformation
    dst = cv2.warpPerspective(src=img, M=M, dsize=(cols, rows));
    
    # visualize and write the new image
    cv2.imshow('Skewed Output', dst)
    
    
def openCvScaling(img):
    # as from https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
    # not implemented yet, just put here since we expect to use it at some point
    res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
    #OR
    height, width = img.shape[:2]
    res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)


######################## 1. IMPORT THE PDF AS AN IMAGE ########################
# See: 
# https://github.com/Belval/pdf2image/blob/master/docs/reference.md
# https://www.geeksforgeeks.org/convert-pdf-to-image-using-python/

# specify the file to read in
## In future versions, will likely have this as an input from the function
folderLoc = 'C:\\Users\\sidne\\OneDrive\\Documents\\2021\\projector patterns project\\'; 
fileLoc = folderLoc + 'croppedCalibration2.pdf';
outputImageLoc = folderLoc + 'croppedCalibration2.jpg';
sidneyPopplerPath = r"C:\Users\sidne\OneDrive\Documents\2021\projector patterns project\poppler\Release-20.12.1\poppler-20.12.1\Library\bin";
 
# read in the pdf as an image (belive it's saved as a ppm image)
pages = convert_from_path(fileLoc, dpi=300, poppler_path = sidneyPopplerPath);

# Save the image as a jpg
## FUTURE VERSION: Import the pdf in a way that you don't have to save it
# IF THE SAVED IMAGE IS BLANK, THE INCOMING PDF IS TOO BIG
pages[0].save(outputImageLoc, 'JPEG', quality=85)

# now load the saved jpg to mess with it
img = cv2.imread(outputImageLoc)


######################## DETECT THE CUTTING AREA (DETECT THE LARGEST RECTANGLE IN VIEW) ########################
originPts = findRectangle(img);



######################## SKEW THE IMAGE (PERSPECTIVE TRANSFORM IN OPENCV) ########################
# https://stackoverflow.com/questions/33497736/opencv-adjusting-photo-with-skew-angle-tilt
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/


outputSkewedLoc = folderLoc + 'croppedCalibration2_skew.jpg';
# display the image to make sure it's accurate
plt.imshow(img)

# perform the perspective transform
perspectiveTransform1(img)

######################## SCALE THE IMAGE ########################



######################## DISPLAY AND/OR SAVE THE IMAGE ########################



#cv2.imwrite(outputSkewedLoc, dst)
#cv2.waitKey()









