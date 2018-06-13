#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:59:36 2018

@author: fulvio
"""

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import cv2
import numpy as np
import math
from sklearn.mixture import GaussianMixture
import scipy.signal as signal
from matplotlib.patches import Ellipse

def concatenateFeatures(rawFeatures_filtered, processedFeatures):
    totalFeatures = np.concatenate((rawFeatures_filtered, processedFeatures), 1)
    return totalFeatures
    

def loadColorImages(folder):
    colorBGR = cv2.imread(folder + "/color.png", 1)
    colorRGB = cv2.cvtColor(colorBGR, cv2.COLOR_BGR2RGB)
    
    hsv = cv2.cvtColor(colorRGB, cv2.COLOR_RGB2HSV)
    luv = cv2.cvtColor(colorRGB, cv2.COLOR_RGB2LUV)
    ycrcb = cv2.cvtColor(colorRGB, cv2.COLOR_RGB2YCrCb)
    
    colorRGB_float = colorRGB.astype(float)
    colorRGB_float = np.divide(colorRGB_float, 254.0)
    hsv = hsv.astype(float)
    luv = luv.astype(float)
    ycrcb = ycrcb.astype(float)
    
    hsv = np.divide(hsv, 255.0)
    luv = np.divide(luv, 255.0)
    ycrcb = np.divide(ycrcb, 255.0)
    
    return colorRGB_float, hsv, luv, ycrcb




def loadNormalsImages(folder):
    normalsBGR = cv2.imread(folder + "/normals.png", 1)
    normalsRGB = cv2.cvtColor(normalsBGR, cv2.COLOR_BGR2RGB)
    
    normalsRGB_float = normalsRGB.astype(float)
    normalsRGB_float = np.divide(normalsRGB_float, 255.0)
    
    
    normalsRGB_float = np.multiply(normalsRGB_float, 2.0)
    normalsRGB_float = np.subtract(normalsRGB_float, 1)
    normalsRGB_float = np.clip(normalsRGB_float, 0, 1)
      
    
    return normalsRGB_float

    
def loadDepthImages(folder):
    depth = cv2.imread(folder + "/depth.png", 0)
    
    depth_float = depth.astype(float)
    depth_float = np.divide(depth_float, 255.0)
    
    return depth_float


def loadSunDir(folder):
    sunDirFile = open(folder + "/sunDir.txt", 'r')
    sunDirFileLines = sunDirFile.readlines()
    sunDirString = sunDirFileLines[0]
    
    sunDirStringValues = sunDirString.split('?')
    sunDir = np.zeros((3), np.float)
    
    sunDir[0] = np.float(sunDirStringValues[0])
    sunDir[1] = np.float(sunDirStringValues[1])
    sunDir[2] = np.float(sunDirStringValues[2])
    
    return sunDir

def processFeatures(rawFeatures):
    # raw features indices:
    # 0: R
    # 1: G
    # 2: B
    # 3: normal X
    # 4: normal Y
    # 5: normal Z
    # 6: depth
    # 7: Hue (first hsv channel)
    # 8: saturation (second hsv channel)
    # 9: U (LUV channel)
    # 10: V (LUV channel)
    # 11: Cr (YCrCb channel)
    # 12: Cb (YCrCb channel)
    
# Processed Features:
    # 0: normal X
    # 1: normal Y
    # 2: normal Z
    # 3: depth
    # 4: log(R/G) = logR
    # 5: log(B/G) = logB
    # 6: (R / R+G+B) = chR
    # 7: (G / R+G+B) = chG
    # 8: (B / R+G+B) = chB
    # 9: Hue (first hsv channel)
    # 10: saturation (second hsv channel)
    # 11: U (LUV channel)
    # 12: V (LUV channel)
    # 13: Cr (LUV channel)
    # 14: Cb (LUV channel)
    processedFeatures = np.zeros((rawFeatures.shape[0], 15), np.float)
    
    for i in range(rawFeatures.shape[0]):
        processedFeatures[i] = fillProcessedFeatureElement(rawFeatures[i])
        
    return processedFeatures

def fillProcessedFeatureElement(rawFeaturesElement):
    
    processedFeaturesElement = np.zeros((15), np.float)

	
    processedFeaturesElement[0] = rawFeaturesElement[3]
    processedFeaturesElement[1] = rawFeaturesElement[4]
    processedFeaturesElement[2] = rawFeaturesElement[5]
    processedFeaturesElement[3] = rawFeaturesElement[6]
    processedFeaturesElement[4] = rgb_to_logR(rawFeaturesElement[0:3])
    processedFeaturesElement[5] = rgb_to_logB(rawFeaturesElement[0:3])
    processedFeaturesElement[6] = rgb_to_chR(rawFeaturesElement[0:3])
    processedFeaturesElement[7] = rgb_to_chG(rawFeaturesElement[0:3])
    processedFeaturesElement[8] = rgb_to_chB(rawFeaturesElement[0:3])
    processedFeaturesElement[9] = rawFeaturesElement[7]
    processedFeaturesElement[10] = rawFeaturesElement[8]
    processedFeaturesElement[11] = rawFeaturesElement[9]
    processedFeaturesElement[12] = rawFeaturesElement[10]
    processedFeaturesElement[13] = rawFeaturesElement[11]
    processedFeaturesElement[14] = rawFeaturesElement[12]
    
    return processedFeaturesElement


    

def rgb_to_logR(rgb):
    return np.log(rgb[0] / rgb[1])
     
def rgb_to_logB(rgb):
    return np.log(rgb[2] / rgb[1])

def rgb_to_chR(rgb):
    return (3 * rgb[0]) / (rgb[0] + rgb[1] + rgb[2])

def rgb_to_chG(rgb):
    return (3 * rgb[1]) / (rgb[0] + rgb[1] + rgb[2])

def rgb_to_chB(rgb):
    return (3 * rgb[2]) / (rgb[0] + rgb[1] + rgb[2])
    
def removeSaturated(rawFeatures):
    newFeatures =  np.zeros((rawFeatures.shape), np.float)
    saturatedIndices = []
    indexNewFeatures = 0
    
    for i in range(rawFeatures.shape[0]):
        if not checkSaturatedRawFeature(rawFeatures[i]):
            newFeatures[indexNewFeatures] = rawFeatures[i]
            indexNewFeatures += 1
        else:
            saturatedIndices.append(i)
            
            
    newFeatures = newFeatures[0:indexNewFeatures, :]
    return newFeatures, saturatedIndices
   
def checkSaturatedRawFeature(rawFeature):
    
    # check if rgb was 255 or 0,
    # rgb are divided by 254 before this check, so 1 corrisponds to a 254 pixel value
    if(rawFeature[0] == 0 or rawFeature[0] > 1 or 
       rawFeature[1] == 0 or rawFeature[1] > 1 or 
       rawFeature[2] == 0 or rawFeature[2] > 1):
        return True
    
    # check if normal values are available
    if(rawFeature[3] == 0 and rawFeature[4] == 0 and rawFeature[5] == 0):
        return True
    if(rawFeature[6] == 0):
        return True
    
    return False
    
def makeArray3(image3channel):
    features1 = makeArray(image3channel[:,:,0])
    features2 = makeArray(image3channel[:,:,1])
    features3 = makeArray(image3channel[:,:,2])
    
    features = np.zeros((features1.shape[0],3))
    
    features[:,0] = features1
    features[:,1] = features2
    features[:,2] = features3
    
    return features
    
def makeArray(image1channel):
    array = image1channel.ravel()
    return array

def concatenateRawFeatures(colorRGB_array, normalsRGB_array, depth_array, hsv, luv, ycrcb):
    ## we create a matrix of features for each pixel
    # raw features indices:
    # 0: R
    # 1: G
    # 2: B
    # 3: normal X
    # 4: normal Y
    # 5: normal Z
    # 6: depth
    # 7: Hue (first hsv channel)
    # 8: saturation (second hsv channel)
    # 9: U (LUV channel)
    # 10: V (LUV channel)
    # 11: Cr (YCrCb channel)
    # 12: Cb (YCrCb channel)
    
    rawFeatures = np.zeros((depth_array.shape[0], 13), np.float)
    
    for i in range(depth_array.shape[0]):
        rawFeatures[i,0] = colorRGB_array[i,0]
        rawFeatures[i,1] = colorRGB_array[i,1]
        rawFeatures[i,2] = colorRGB_array[i,2]
        rawFeatures[i,3] = normalsRGB_array[i,0]
        rawFeatures[i,4] = normalsRGB_array[i,1]
        rawFeatures[i,5] = normalsRGB_array[i,2]
        rawFeatures[i,6] = depth_array[i]
        rawFeatures[i,7] = hsv[i,0]
        rawFeatures[i,8] = hsv[i,1]
        rawFeatures[i,9] = luv[i,1]
        rawFeatures[i,10] = luv[i,2]
        rawFeatures[i,11] = ycrcb[i,1]
        rawFeatures[i,12] = ycrcb[i,2]
        
    return rawFeatures


def computeGaussSimilarity(mean1, mean2, sigma1, sigma2):

    crossCorrInZero = np.multiply( mlab.normpdf(np.zeros((1), np.float), mean1 - mean2, math.sqrt(sigma1**2 + sigma2**2)), math.sqrt(sigma1**2 + sigma2**2) * math.sqrt(math.pi * 2) )
    return crossCorrInZero






def bgr_to_chG(bgrImg):
    rows, cols, _ = bgrImg.shape
    
    chR = np.zeros((rows, cols), np.float)
    chG = np.zeros((rows, cols), np.float)
    chB = np.zeros((rows, cols), np.float)
    
    for x in range(0,rows):
        for y in range(0,cols):
            if  (bgrImg[x,y,1] != 0):
               
                r = float(bgrImg[x,y,2]) / float(255)
                g = float(bgrImg[x,y,1]) / float(255)
                b = float(bgrImg[x,y,0]) / float(255)
                
                
                #g chromaticity component
                chG[x,y] = np.uint8(g*255 / (r+g+b))
                
    return chG



def bgr_to_chR_chG(bgrImg):
    rows, cols, _ = bgrImg.shape
    
    chR = np.zeros((rows, cols), np.float)
    chG = np.zeros((rows, cols), np.float)
    chB = np.zeros((rows, cols), np.float)
    
    for x in range(0,rows):
        for y in range(0,cols):
            if  (bgrImg[x,y,1] != 0):
               
                r = float(bgrImg[x,y,2]) / float(255)
                g = float(bgrImg[x,y,1]) / float(255)
                b = float(bgrImg[x,y,0]) / float(255)
                
                # r chromaticity componen
                chR[x,y] = np.uint8(r*255 / (r+g+b))
                
                #g chromaticity component
                chG[x,y] = np.uint8(g*255 / (r+g+b))
                
                chB[x,y] = np.uint8(b*255 / (r+g+b))
                
    return chR, chB, chG




#img = cv2.imread('RGB_40.png')
##hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
#chR, chB, chG = bgr_to_chR_chG(img)
#

#
#
#gmm = GaussianMixture(n_components).fit(X)
#prob = gmm.predict_proba(X)
##plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

#
#plt.figure()
#plt.imshow(probClasses, cmap='gray')
#plt.title("probabilities")
#
#plt.figure()
#plt.imshow(indexClasses, cmap='nipy_spectral')
#plt.title("classes")
#
#
#probClasses
#cv2.imwrite("probabilities.png", probClasses)
#
##labels = labels.reshape((chR.shape[0],chB.shape[1]))
##plt.figure()
##labels = np.multiply(labels, 255/n_components)
##plt.imshow(labels, cmap='gray')
#
##chR = chR.ravel()
###chG = chG.ravel()
##chB = chB.ravel()
##
#
##
##r = cv2.selectROI(img[:,:,0])
###hCrop = h[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
###lCrop = l[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
##chBCrop = chB[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
##chRCrop = chR[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
##chBCrop = chBCrop.ravel()
##chRCrop = chRCrop.ravel()
##
###
#
#
##chromaticity = np.zeros((chR.shape[0], 2), np.float)
##chromaticity[:,0] = chR
##chromaticity[:,1] = chB
#
##fft2 = fftpack.fft2(chB)
##plt.imshow(np.log10(abs(fft2)))
##plt.show()
##
####
###chromaticity = np.divide(chromaticity, 255)
###
##plt.scatter(chRCrop, chBCrop)

#print("Fulvio is a penis")