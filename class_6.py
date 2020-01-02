import numpy as np
import cv2 as cv


# 模糊操作

def blur_demo(image):
    dst = cv.blur(image, (10, 10))  # 后面是二维的ksize 卷积核 (x方向，y方向）定义卷积核
    cv.imshow('img', dst)


img = cv.imread('./WindowsLogo.jpg', 1)  # blue green red
blur_demo(img)
t1 = cv.getTickCount()

t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
