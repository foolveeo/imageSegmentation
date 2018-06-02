#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:48:34 2018

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

sigma1 = 12
sigma2 = 12

mean1 = 43
mean2 = 4

x = np.linspace(-30,30, 1000)
normValues1 = np.multiply(mlab.normpdf(x, 0, sigma1), 1)
normValues2 = np.multiply(mlab.normpdf(x, mean2-mean1, sigma2), 1)

plt.figure()
plt.plot(x,normValues1, label="1")
plt.plot(x, normValues2, label="2")
plt.show()


x = np.linspace(-30,30, 1000*2-1)
plt.figure()
#crossCorr = np.divide(signal.convolve(normValues1, normValues2), (sigma1 + sigma2) * math.pi * 4 )
crossCorr = np.multiply(mlab.normpdf(x, mean1 - mean2, math.sqrt(sigma1**2 + sigma2**2)),  math.sqrt(sigma1**2 + sigma2**2) * math.sqrt(math.pi * 2))
print(crossCorr[999])

plt.plot(x, crossCorr)
plt.legend()
plt.show()
        
