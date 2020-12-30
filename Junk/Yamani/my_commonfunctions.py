

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import *

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb

import cv2


# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

# Show the figures / plots inside the notebook
def my_show_images(images,titles=None, row_max=1, dpi=200):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure(dpi=dpi)
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(math.ceil(n_ims/row_max), row_max, n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims/row_max)
    plt.show() 
    

def my_show_hist(img):
    img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    hist = np.histogram(img, range(0, 257))
    # print(hist)
    plt.bar(hist[1][:-1], hist[0])
    plt.show()


def my_imread_gray(fname):
    img = io.imread(fname)
    if len(img.shape) == 2:
        return img
    elif img.shape[2] == 4:
        return rgb2gray(rgba2rgb(img))
    else:
        return rgb2gray(img)


def my_close(src, kernel):
    dilated = cv2.dilate(src, kernel)
    return cv2.erode(dilated, kernel)


def my_open(src, kernel):
    dilated = cv2.erode(src, kernel)
    return cv2.dilate(dilated, kernel)


def get_distance_between_staves(img_binary_bg_white):
    img_height = img_binary_bg_white.shape[0]
    
    flattened = img_binary_bg_white.T.flatten()
    flattened_indices = np.arange(0, flattened.shape[0], 1, np.uint32)
    flattened[flattened_indices % img_height == 0] = False # Separate each column with a black pixel
    
    image, contours, hierarchy = cv2.findContours((flattened*255).astype(np.uint8), 
                                              cv2.RETR_TREE, 
                                              cv2.CHAIN_APPROX_SIMPLE)
    
    # We refer to length as the vertical distance between 2 black pixels
    length_freq = np.zeros((img_height), dtype=np.uint32) # No contour can be taller than img_height because we separated each column with a black pixel
    all_possible_lengths = np.arange(0, img_height, 1, dtype=np.uint32)
    for i in contours:
        contour_y = i.T[1]
        length = contour_y[0][1] - contour_y[0][0] if len(contour_y[0]) == 2 else 1
        length_freq[length] += 1
        
    return all_possible_lengths[length_freq == length_freq.max()][0]


def get_line_separation_kernel_size_from_distance_between_staves(distance_between_staves):
    if distance_between_staves % 2 == 0:
        return distance_between_staves + 9
    else:
        return distance_between_staves + 8