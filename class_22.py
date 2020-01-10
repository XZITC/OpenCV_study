import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 二值图像形态学操作  膨胀 腐蚀 开 闭 顶帽 礼帽
capture = cv.VideoCapture(1)
capture.set(3, 1280)
capture.set(4, 720)


def resize_func(image):
    dst = cv.resize(img, None, fx=0.28, fy=0.28, interpolation=cv.INTER_CUBIC)
    # cv.imwrite('resize.jpg', dst)
    return dst


# canny
def canny_demo(image):
    dst = resize_func(image)
    blur = cv.GaussianBlur(dst, (5, 5), 0)  # 高斯模糊去噪声
    edge_canny_one = cv.Canny(blur, 45, 150)  # 小波边缘检测？？？
    cv.imshow('canny', edge_canny_one)
    return edge_canny_one


def dilate_demo(image):
    binary_img = canny_demo(image)
    # 获取核函数 cv.MORPH_RECT矩形的核函数 anchor默认在中心
    # cv.MORPH_ELLIPSE 椭圆型核函数
    kernel_ele = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    print(kernel_ele)
    kernel = np.ones([3, 3], np.uint8)
    # kernel = np.array([[1],[1],[1]],np.uint8) #一维kernel，膨胀本质就是或操作
    dst = cv.dilate(binary_img, kernel, iterations=1)
    cv.imshow('dilate_img', dst)


def more_phology(image):  # 多形态学运算
    # 开运算 先腐蚀再膨胀 先瘦再胖
    binary_img = canny_demo(image)
    kernel = np.ones([5, 5], np.uint8)
    # opening = cv.morphologyEx(binary_img,cv.MORPH_OPEN,kernel)
    # cv.imshow('opening', opening)
    # 开运算 先膨胀后腐蚀 先胖再瘦
    closing = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel)
    cv.imshow('closing', closing)


def erdor_demo(image):  # 腐蚀操作
    binary_img = canny_demo(image)
    kernel = np.ones([3, 3], np.uint8)  # 二位kernel 3，3里操作：与 都是1才1
    dst = cv.erode(binary_img, kernel, iterations=1)
    cv.imshow('erdor_img', dst)


def binary_jiangzao(image):
    binary_img = canny_demo(image)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dil = cv.dilate(binary_img, kernel, iterations=1)
    cv.imshow('dilate_img', dil)


img = cv.imread('./image/win.jpg', 1)  # blue green red
t1 = cv.getTickCount()
binary_jiangzao(img)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
