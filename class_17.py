import numpy as np
import cv2 as cv
from matplotlib import pyplot  as plt


def canny_demo(image):
    """
    canny过程：1.高斯滤波
    2.图像梯度计算 梯度就用Sobel算子计算 相加开方，计算梯度tan = Gx/Gy
    3.非极大值抑制 筛选出局部最大的梯度值
    4.双阈值处理 一小一大，比值通常1：3或1：2
    """
    blur = cv.GaussianBlur(image,(3,3),0) #高斯模糊去噪声
    gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
    # x方向梯度
    grad_x = cv.Sobel(gray,cv.CV_16SC1,1,0)
    # y方向梯度
    grad_y = cv.Sobel(gray,cv.CV_16SC1,0,1)
    #dege 后面两个比值一般是3：1或者2：1
    edge_canny_one = cv.Canny(blur,50,150)
    # cv.canny:p1:8bit的图像，就uint8型 p2:
    edge_canny = cv.Canny(grad_x,grad_y,50,100)
    cv.imshow('Canny_dege',edge_canny_one)


img = cv.imread('./image/load_sunny.jpg', 1)  # blue green red
#video_soble()
canny_demo(img)
t1 = cv.getTickCount()
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()