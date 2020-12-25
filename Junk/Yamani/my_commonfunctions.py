

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

