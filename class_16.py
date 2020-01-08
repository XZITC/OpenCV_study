import numpy as np
import cv2 as cv
from matplotlib import pyplot  as plt

#边缘提取，Sobel lapalain

def sobel_demo(image):
    grad_x = cv.Sobel(image,cv.CV_32F,1,0) #在x方向上求梯度
    grad_y = cv.Sobel(image,cv.CV_32F,0,1)#在y方向上求梯度
    grad_force_x = cv.Scharr(image,cv.CV_32F,1,0) #Scharr加强版
    gradx = cv.convertScaleAbs(grad_x) #需要加绝对值 sobel负的边缘变正号
    grady = cv.convertScaleAbs(grad_y)
    final = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow('11',gradx)
    cv.imshow('22',grady)
    cv.imshow('33',grad_force_x)

def video_soble():
    capture = cv.VideoCapture(0)
    while (True):
        ret, frame = capture.read()
        # frame = cv.flip(frame, -1)  # cv.filp()镜像变换 1：左右 -1上下
        grad_x = cv.Sobel(frame, cv.CV_32F, 1, 0)  # 在x方向上求梯度
        grad_y = cv.Sobel(frame, cv.CV_32F, 0, 1)  # 在y方向上求梯度
        gradx = cv.convertScaleAbs(grad_x)
        grady = cv.convertScaleAbs(grad_y)
        final = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
        cv.imshow('rawVideo', final)
        if cv.waitKey(60) == 32:  # 32 ascii空格
            capture.release()
            cv.destroyAllWindows()
            break



img = cv.imread('./image/load_sunny.jpg', 1)  # blue green red
#video_soble()
sobel_demo(img)
t1 = cv.getTickCount()
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()