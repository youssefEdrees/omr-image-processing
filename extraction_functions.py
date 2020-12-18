#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from commonfunctions import *


#cv.imshow("Source", src)
#cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
#cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)


# In[ ]:





# In[ ]:





# In[3]:


def find_4_points(img):
    img = np.copy(img)
    img = cv.Canny(img, 50, 200, None, 3)
    
    pts = np.transpose((np.nonzero(img)[1],np.nonzero(img)[0]))
    #pts = np.non(pts)
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
    


# In[24]:


def fix_skew_and_rotation(original_img):
    
    img = smooth_image(original_img)
    #img = make_horizontal(img)
    points = find_4_points(img)
    
    diameter1 = math.sqrt((points[0][0]-points[2][0])**2+(points[0][1]-points[2][1])**2)
    diameter2 = math.sqrt((points[1][0]-points[3][0])**2+(points[1][1]-points[3][1])**2)
    
    direction1 = points[2]-points[0]
    direction2 = points[3]-points[1]
    
    points[0] = points[0] - 0.07 * direction1
    points[2] = points[2] + 0.07 * direction1
    
    points[1] = points[1] - 0.07 * direction2
    points[3] = points[3] + 0.07 * direction2
    
    
    diameter1 = math.sqrt((points[0][0]-points[2][0])**2+(points[0][1]-points[2][1])**2)
    diameter2 = math.sqrt((points[1][0]-points[3][0])**2+(points[1][1]-points[3][1])**2)
    
    direction1 = points[2]-points[0]
    direction2 = points[3]-points[1]
    
    
    d = max(diameter1,diameter2)
    
    unit_direction1 = direction1 / np.linalg.norm(direction1)
    unit_direction2 = direction2 / np.linalg.norm(direction2)
    
    dot_product = np.dot(unit_direction1, unit_direction2)
    
    angle_between = np.arccos(dot_product)
    angle = (math.pi-angle_between)/2
    maxWidth  = round( d * math.cos(angle))
    maxHeight = round(d * math.sin(angle))
    
    print(maxWidth)
    print(maxHeight)
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(points, dst)
    warped = cv.warpPerspective(original_img, M, (maxWidth, maxHeight),borderValue=(255,255,255))
    # return the warped image
    return (warped)
    
        


# In[5]:


def make_horizontal(img):
    
    dst = cv.Canny(img, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2RGB)
    cdstP = np.copy(cdst)
   
    linesP = cv.HoughLinesP(dst, 4, np.pi / 180, 50, None, 50, 10)
    if(linesP is None):
        return img
    l = linesP[0][0]
    
    angle = math.degrees(math.atan2(l[1]-l[3],l[0]-l[2]))
    print(l)
    print(angle)
    
    if angle < 0 :
        angle = -1 * angle
    if (angle > 90):
        angle = -1 * (180-angle)
    return rotate_bound(img,angle)
    

    
    
    
    
    
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])


    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))


    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))


# In[7]:



dsaasfda 
default_file = 'test_rotate.png'

src_skew = cv.imread('test_rotate.png', cv.IMREAD_GRAYSCALE)
src_sheer = cv.imread('test_sheer.png', cv.IMREAD_GRAYSCALE)

skew_with_points = np.copy(src_skew)
points = find_4_points(skew_with_points)        
for i in range(4):
    skew_with_points = cv.circle(skew_with_points,(points[i][0],points[i][1]), radius=10, color=(0, 0, 255), thickness=-1)

sheer_with_points = np.copy(src_sheer)
points = find_4_points(sheer_with_points)        
for i in range(4):
    sheer_with_points = cv.circle(sheer_with_points,(points[i][0],points[i][1]), radius=10, color=(0, 0, 255), thickness=-1)


x1  = fix_skew_and_rotation(src_skew)
x2  = fix_skew_and_rotation(src_sheer)

show_images([src_skew,skew_with_points,x1])
show_images([src_sheer,sheer_with_points,x2])
"""
dst = cv.Canny(x, 50, 200, None, 3)

# Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2RGB)
cdstP = np.copy(cdst)

lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)


linesP = cv.HoughLinesP(dst, 4, np.pi / 180, 50, None, 50, 10)


l = linesP[len(linesP)-1][0]
cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)


if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
"""



#points = find_4_points(src)        
#for i in range(4):
#    src = cv.circle(src,(points[i][0],points[i][1]), radius=10, color=(0, 0, 255), thickness=-1)







# In[22]:


tests = []
for  i in range(1,11):
    tests.append(cv.imread('tests/test_'+str(i)+'.jpg', cv.IMREAD_GRAYSCALE))


# In[28]:


def smooth_image(img):
    img = cv.GaussianBlur(img, (5,5) , 2)
    
    cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,7,2)
    
    structuring_element = np.ones((7,7))
    img = cv.medianBlur(img,7)
    img = cv.erode(img , structuring_element)
    img = cv.dilate(img , structuring_element)
    
    
    #img = cv.erode(img , structuring_element)
    #img = cv.dilate(img , structuring_element)
    #img = cv.erode(img , structuring_element)
    #img = cv.dilate(img , structuring_element)
    
    #blur = cv.GaussianBlur(img,(7,7),4)
    #ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #img = cv.medianBlur(img,7)
    
    
    return img


# In[29]:


for  i in range(10):
    show_images([tests[i],draw_points(smooth_image(tests[i])),fix_skew_and_rotation(tests[i])])


# In[ ]:


kk = binarize_image(tests[0])
show_images([kk])


# In[26]:


def draw_points(img):
    points = find_4_points(img)        
    img2 = np.copy(img)
    for i in range(4):
        img2 = cv.circle(img,(points[i][0],points[i][1]), radius=10, color=(0,0, 0), thickness=-1)
    return img2

