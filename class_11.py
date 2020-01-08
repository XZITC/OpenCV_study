import cv2 as cv
import numpy as np
from matplotlib import pyplot  as plt


def back_project_demo(roi_img,raw_img): #直方图方向查找
    hsv = cv.cvtColor(roi_img, cv.COLOR_BGR2HSV)
    raw_hsv = cv.cvtColor(raw_img, cv.COLOR_BGR2HSV)
    roi_hist = cv.calcHist([hsv], [0, 1], None, [400, 400], [0, 180, 0, 256])
    cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
    dst = cv.calcBackProject([raw_hsv],[0,1],roi_hist,[0, 180, 0, 256],1)
    cv.imshow('backprojrct_demo',dst)


# 直方图反向投影
def hist2d_demo(image):
    """
    2d直方图的制作：（x：s  y：h）
    1.转换到hsv色彩空间
    2.cv.calHist统计直方图 p1：img必须方括号
    p2：用什么通道
    p3：遮罩
    p4：多少哥直方柱，或者理解为x轴或y轴坐标的尺度比例
    p5：ranges参数表示像素值的范围，通常[0,256]。
        此外，假如channels为[0,1],ranges为[0,256,0,180],则代表0通道范围是0-256,1通道范围0-180。

    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(hist,interpolation='nearest')
    #cv.imshow('hist2d_demo',hist)


img = cv.imread('./image/load_sunny.jpg', 1)  # blue green red
img_roi = cv.imread('./image/load_sunny_roi.jpg', 1)
back_project_demo(img_roi,img)
# #plt.xlim([0, 256])
#cv.imshow('img', img)
#hist2d_demo(img)
