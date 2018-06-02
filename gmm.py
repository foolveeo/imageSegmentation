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


def computeGaussSimilarity(mean1, mean2, sigma1, sigma2):

    crossCorrInZero = np.multiply( mlab.normpdf(np.zeros((1), np.float), mean1 - mean2, math.sqrt(sigma1**2 + sigma2**2)), math.sqrt(sigma1**2 + sigma2**2) * math.sqrt(math.pi * 2) )
    return crossCorrInZero



def draw_ellipse(position, covariance, ax=None, **kwargs): 
    """Draw an ellipse with a given position and covariance""" 
    ax = ax or plt.gca()
        
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0])) 
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
        # Draw the Ellipse
    
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
    

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2) 
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
        ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)


plt.close("all")


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

def GMMChromaticityGreen(imgPath, nbr_classes, min_prob):
    
    
    img = cv2.imread(imgPath)
    chG = bgr_to_chG(img)
    
    
    plt.figure()
    plt.subplot(111) 
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("RGB")
    
    
    X = np.zeros((chG.shape[0]*chG.shape[1],1))
    
    X[:,0] = chG.ravel()
    
    gmm = GaussianMixture(nbr_classes).fit(X)
    prob = gmm.predict_proba(X)
    
    
    x = np.linspace(0,256, 2560)
    means = gmm.means_
    covariances = gmm.covariances_
    sigmas = np.sqrt(covariances)
    normValues = np.zeros((nbr_classes, x.shape[0]), np.float)
    
#    plt.figure()
    for i in range(nbr_classes):  
        normValues[i] = mlab.normpdf(x, means[i], sigmas[i])
#        plt.plot(x, normValues[i], label=str(i))
#    plt.legend()
#    plt.show()
    
    crossCorr = np.zeros((nbr_classes, nbr_classes), np.float)
    
    
    #x = np.linspace(-255, 255, 2*2560-1)
    
    #for i in range(nbr_classes):
        #plt.figure()
        #for j in range(nbr_classes):
            #crossCorr[i,j] = signal.correlate(normValues[i], normValues[j])
            
            #plt.plot(x, crossCorr[i,j], label=(str(i) + " - " + str(j)))
        #plt.legend()
        #plt.show()
        
    #plt.figure()
    
#    real_nbr_classes = nbr_classes
#    for i in range(nbr_classes):
#        for j in range(nbr_classes):
#            if i > j:
#                crossCorr[i,j] = computeGaussSimilarity(means[i], means[j], sigmas[i], sigmas[j])
#                if crossCorr[i,j] > 0.6:
#                    prob[:,i] = prob[:,i] + prob[:,j]
#                    prob[:,j] = np.zeros(prob[:,j].shape)
#                    real_nbr_classes -= 1
#                    
#                
#    print("from ", nbr_classes, " to ", real_nbr_classes, " classes")
#    plt.figure()
#    plt.imshow(crossCorr, cmap="gray")
#    
#    
    
    indexBestClasses = np.zeros(prob.shape[0])
    bestClassProb = np.zeros(prob.shape[0])
    secondBestClassProb = np.zeros(prob.shape[0])
    indexClasses = np.zeros(prob.shape[0])
    
    for i in range(prob.shape[0]):
        bestClassProb[i] = np.amax(prob[i,:])
        secondBestClassProb[i] = np.sort(prob[i,:])[-1]
        indexClasses[i] = np.argmax(prob[i,:])
        if(bestClassProb[i] < min_prob):
            indexBestClasses[i] = -1
        else:
            indexBestClasses[i] = np.argmax(prob[i,:])
#        if(probClasses[i] < min_prob or min_prob * probClasses[i] < secondBestClassProb):
#            indexBestClasses[i] = -1
#        else:
#            indexBestClasses[i] = np.argmax(prob[i,:])
    
    bestClassProb = bestClassProb.reshape(chG.shape[0],chG.shape[1])
    indexBestClasses = indexBestClasses.reshape(chG.shape[0],chG.shape[1])
    secondBestClassProb = secondBestClassProb.reshape(chG.shape[0],chG.shape[1])
    indexClasses = indexClasses.reshape(chG.shape[0],chG.shape[1])
    
    
#    plt.figure()
#    plt.imshow(bestClassProb, cmap='gray')
#    plt.title("probabilities")
#    
#    plt.figure()
#    plt.imshow(indexClasses, cmap='nipy_spectral')
#    plt.title("classes")
#    
#    plt.figure()
#    plt.imshow(secondBestClassProb, cmap='gray')
#    plt.title("secondProbabilities")
    
    plt.figure()
    plt.imshow(indexBestClasses, cmap='nipy_spectral')
    plt.title("classes selected")
    plt.savefig("classes_" + imgPath)

    
    
    
    
    
    #cv2.imwrite("probabilities.png", probClasses)


GMMChromaticityGreen('RGB_40.png', 4, 0.7)
GMMChromaticityGreen('rgb1239.png', 5, 0.8)
GMMChromaticityGreen('rgb1404.png', 5, 0.8)
#GMMChromaticityGreen('RGB_29.png', 7, 0.7)
#GMMChromaticityGreen('RGB_151.png', 7, 0.7)
#GMMChromaticityGreen('RGB_210.png', 7, 0.7)
#GMMChromaticityGreen('RGB_213.png', 7, 0.7)
#GMMChromaticityGreen('IMG_2106.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2107.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2108.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2109.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2110.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2111.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2113.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2114.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2115.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2116.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2117.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2118.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2119.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2120.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2121.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2122.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2123.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2124.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2125.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2126.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2127.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2128.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2129.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2130.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2131.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2132.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2133.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2134.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2135.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2136.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2137.jpg', 7, 0.7)
#GMMChromaticityGreen('IMG_2138.jpg', 7, 0.7)
#
#
#img = cv2.imread('RGB_40.png')
##hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
#chR, chB, chG = bgr_to_chR_chG(img)
#
#chromaticity = np.zeros((chR.shape[0], chR.shape[1], 3), np.uint8)
#chromaticity[:,:,0] = chR
#chromaticity[:,:,1] = chG
#chromaticity[:,:,2] = chB
#
#
#plt.figure()
#plt.subplot(111) 
#plt.imshow(chromaticity)
#plt.title("RGB")
#
#
#n_components = 4
#
#X = np.zeros((chR.shape[0]*chB.shape[1],1))
#
#X[:,0] = chG.ravel()
##X[:,1] = chB.ravel()
##X[:,2] = chR.ravel()
#
#gmm = GaussianMixture(n_components).fit(X)
#prob = gmm.predict_proba(X)
##plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
#indexClasses = np.zeros(prob.shape[0])
#probClasses = np.zeros(prob.shape[0])
#for i in range(prob.shape[0]):
#    probClasses[i] = np.amax(prob[i,:])
#    if(probClasses[i] < 0.9):
#        indexClasses[i] = -1
#    else:
#        indexClasses[i] = np.argmax(prob[i,:])
#    
#plt.figure()
#plt.hist(probClasses,200,[0,1])
#plt.title("probabilities")
#
#probClasses = probClasses.reshape(chR.shape[0],chB.shape[1])
#indexClasses = indexClasses.reshape(chR.shape[0],chB.shape[1])
#
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