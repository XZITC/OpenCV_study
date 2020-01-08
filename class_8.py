import numpy as np
import cv2 as cv


def bi_demo(img):  # 边缘保留滤波
    # p2:d取0就行 p3：sigmaColor 取的越大，小的差异越会被抹去 p4：sigmaSpace空间差异取小
    dst = cv.bilateralFilter(img, 0, 100, 5)
    dst2 = cv.bilateralFilter(img, 0, 100, 100)
    cv.imshow('bi_demo', dst)
    cv.imshow('bi_demo_50', dst2)


def shift_demo(img):  # 均值迁移滤波
    dst = cv.pyrMeanShiftFiltering(img, 10, 50)
    cv.imshow('shift_demo', dst)


img = cv.imread('./5.jpg', 1)  # blue green red

t1 = cv.getTickCount()
shift_demo(img)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
