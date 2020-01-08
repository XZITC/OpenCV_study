import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# 轮廓发现
# 不必用edge 求梯度后，求出梯度总和，用梯度自动阈值和二值化，从而避免求边缘的时候阈值带来的烦恼
# 比求完梯度后求边缘得出结果要好

def canny_demo(image):
    blur = cv.GaussianBlur(image,(3,3),0) #高斯模糊去噪声
    gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
    # x方向梯度
    grad_x = cv.Sobel(gray,cv.CV_16SC1,1,0)
    # y方向梯度
    grad_y = cv.Sobel(gray,cv.CV_16SC1,0,1)
    #dege 后面两个比值一般是3：1或者2：1
    edge_canny_one = cv.Canny(blur,30,150)
    # cv.canny:p1:8bit的图像，就uint8型 p2:
    #cv.imshow('Canny_dege',edge_canny_one)
    return edge_canny_one


def contours_demo(image):
    """
    image_G = cv.GaussianBlur(img, (0, 0), 1)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转换灰度图

    dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 25, 10)

    """
    dst = canny_demo(image)
    #cv.imshow('binary', dst)
    # p1:二值图像
    # p2：构建模型的方法 cv.RETR_TREE(树形结构)  cv.RETR_EXTERNAL(最大层轮廓)
    contours, heriachy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   # contours 类型 list型
    for i, contour in enumerate(contours):  # 枚举 contours里面是枚举，i：index 对应contour
        # p1:原始图
        # p2:对应轮廓线
        # p3：对应的index 第几个
        # p4：轮廓线颜色
        # p5：绘制线粗细 -1:填充轮廓
        # 踩坑了注意是contours,drawContours通过index下标去找对应的contour
        cv.drawContours(image, contours, i, (0, 100, 255),1)
        #print(i)
        #print(contour)
    # 法二 直接 p3：-1绘制全部 不然就是绘制对应的第几个轮廓
    # all_contours = cv.drawContours(image,contours,-1,(0,0,255))
    cv.imshow('contour_image', image)

img = cv.imread('./image/load_sunny.jpg', 1)  # blue green red
t1 = cv.getTickCount()
contours_demo(img)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()