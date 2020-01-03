import numpy as np
import cv2 as cv
from matplotlib import pyplot  as plt


# 直方图 hisgrom

def his_demo(image):
    # p1:这个参数是指定每个bin(箱子)分布的数据,对应x轴一维数组
    # p2：bins条数，条数越多越细也，就是总共有几条条状图
    # p3:给定的范围图片就是0-255
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

def color_hist(image): #自己单独取各个通道，叠加在一个finger中绘制，颜色通过.hist函数给定
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    plt.hist(b.ravel(), 256, [0, 256],color='blue')
    plt.hist(g.ravel(), 256, [0, 256],color='green')
    plt.hist(r.ravel(), 256, [0, 256],color='red')

def offical_color_demo(image): #官方法绘制直方图
    color =  ('blue','green','red')
    for i ,color in enumerate(color):
        hist = cv.calcHist([image],[i],None,[256],[0,256])
        plt.plot(hist,color = color)
        plt.xlim([0,256])
    plt.show()


img = cv.imread('./WindowsLogo.jpg', 1)  # blue green red
# his_demo(img)
color_hist(img)
#offical_color_demo(img)
# cv.imshow('img', img)
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
