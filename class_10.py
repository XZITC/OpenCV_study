import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def equalHist_demo(image):# 直方图均衡化 仅针对灰度图 自动增加对比度，增强图像
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY) #必须灰度图
    dst = cv.equalizeHist(gray)
    cv.imshow('equalHist',dst)

def calhe_demo(image): #区域均衡化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # p1:最低阈值 p2:块大小
    calhe = cv.createCLAHE(clipLimit=5,tileGridSize=(5,5))
    dst = calhe.apply(gray)
    cv.imshow('calhe_demo',dst)

img = cv.imread('./WindowsLogo.jpg', 1)  # blue green red
calhe_demo(img)