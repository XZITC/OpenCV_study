import cv2 as cv
import numpy as np
#图像读取以及填充

def inverse(image):
    # dest = cv.bitwise_not(image)
    # cv.imshow('img', dest)
    m1 = np.zeros([3, 4], np.uint32)  # 类型 u8 32 64 float
    m1.fill(1222)
    print(m1)
    m2 = np.ones([3, 4], np.float)
    m2.fill(233.434)
    print(m2)


img = cv.imread('C:/Users/XZITC/Desktop/2.jpg', 1)  # blue green red
t1 = cv.getTickCount()
print(img[:, 0, 0])
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
# waitkey()很重要的函数，等待键盘输入事件时间 ord()获取AsCII码
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
