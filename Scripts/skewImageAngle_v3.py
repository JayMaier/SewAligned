# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:29:30 2021

@author: sidne
"""

###########################################
# Sidney CB
# Created: 1/1/2021 (happy new year!)
# DescriptionFirst version of program that will read in a pdf document and 
# skew/tilt it based on a specified angle

# last edited: 1/2/2021
# Changes: v3 - changed structure to be more realistic of running
###########################################

# import necessary packages
from pdf2image import convert_from_path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


def drawBox(image, marker):
    # draw a bounding box around the image and display it
	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.imshow("highlighted image", image);


def findRectangle(image):
    # function to find a rectangle in the image
    # returns the coordinates of the four corners
    
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
    
    drawBox(image, rect)
    
    return box

    
def openCvScaling(img):
    # as from https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
    # not implemented yet, just put here since we expect to use it at some point
    res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    #OR
    height, width = img.shape[:2]
    res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)



# Set up locations
folderLoc = 'C:\\Users\\sidne\\OneDrive\\Documents\\2021\\projector patterns project\\'; 
rectImageLoc = folderLoc + '3ft.jpg';

fileLoc = folderLoc + 'croppedCalibration2.pdf';
outputImageLoc = folderLoc + 'croppedCalibration2.jpg';
outputSkewedLoc = folderLoc + 'croppedCalibration2_skew.jpg';

sidneyPopplerPath = r"C:\Users\sidne\OneDrive\Documents\2021\projector patterns project\poppler\Release-20.12.1\poppler-20.12.1\Library\bin";


######################## 1. READ IN IMAGE OF CUTTING SURFACE ########################
## FUTURE VERSION: get image from webcam rather than a saved jpg
rectImg = cv2.imread(rectImageLoc)

# visualize the image
cv2.imshow('Cutting Surface Skewed Input', rectImg)

# find the coordinates of the cutting board corners
# save these as the origin points for the followingperspective transform
rectOriginPts = findRectangle(rectImg)


######################## 2. SQUARE IMAGE OF CUTTING SURFACE USING PERSPECTIVE TRANSFORM ########################
# perform the perspective transform
newCols = 800; newRows = 600; # use reasonable size of python screen for now
rectTargetPts = np.float32(
    [[0, newRows],
     [0, 0],
     [newCols, 0],
     [newCols, newRows]]
)    

# get the perspective transform
M = cv2.getPerspectiveTransform(rectOriginPts, rectTargetPts)
# warp the image using the transform matrix
dst = cv2.warpPerspective(src=rectImg, M=M, dsize=(newCols, newRows));
# visualize the new image
cv2.imshow('Cutting Surface Square Output', dst)

######################## 3. IMPORT THE PDF OF A PATTERN ########################
# https://github.com/Belval/pdf2image/blob/master/docs/reference.md
# https://www.geeksforgeeks.org/convert-pdf-to-image-using-python/

## In future versions, will likely have the pdf location as an input from the function

# read in the pdf, convert to an image
pages = convert_from_path(fileLoc, dpi=300, poppler_path = sidneyPopplerPath);

# Save the image as a jpg
## FUTURE VERSION: Import the pdf in a way that you don't have to save it
## NOTE: IF THE SAVED IMAGE IS BLANK, THE INCOMING PDF IS TOO BIG
pages[0].save(outputImageLoc, 'JPEG', quality=85)

# now load the saved jpg to mess with it
patternImg = cv2.imread(outputImageLoc)


######################## 4. SKEW THE PATTERN IN THE SAME WAY AS THE CUTTING BOARD IMAGE WAS ########################
# https://stackoverflow.com/questions/33497736/opencv-adjusting-photo-with-skew-angle-tilt
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/


# display the image to make sure it's accurate
cv2.imshow('Pattern input', patternImg)

# use the cutting board origin points as the pdf target points
pdfTargetPts = rectOriginPts.copy();

# assume that all aspects of the pdf pattern are relevant
# use the corners of the image as the pattern origin points
rows, cols, ch = patternImg.shape
pdfOriginPts = np.float32(
    [[0, 0],
     [cols, 0],
     [0, rows],
     [cols, rows]]
 )   

# get the perspective transform
M = cv2.getPerspectiveTransform(pdfOriginPts, pdfTargetPts)
# warp the image using the transform matrix
patternDst = cv2.warpPerspective(src=patternImg, M=M, dsize=(cols, rows));

# visualize and write the new image
cv2.imshow('Pattern Skewed Output', patternDst)


######################## SCALE THE IMAGE ########################



######################## DISPLAY AND/OR SAVE THE IMAGE ########################



#cv2.imwrite(outputSkewedLoc, dst)
#cv2.waitKey()









