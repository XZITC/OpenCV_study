import numpy as np
import cv2 as cv


# 模糊操作

def blur_demo(image):  # 均值模糊 去噪声
    dst = cv.blur(image, (10, 10))  # 后面是二维的ksize 卷积核 (x方向，y方向）定义卷积核
    cv.imshow('img', dst)


def mid_blur_demo(image):
    dst = cv.medianBlur(image, 5)  # 中值模糊，去椒盐噪声 黑白颗粒
    cv.imshow('medianBlur', dst)


def custom_blur_demo(image):
    """
    卷积实质就是一个矩阵对一个区域各项累加，去噪声好比把凹凸不平的第填平
    通过卷积核求得平均值达到去噪声的目的

    kernel定义原则 1.奇数 2.相加总和为0：边缘梯度 1：增强
    """
    kernel = np.ones([5, 5], np.float32) / 25  # 最大情况每个点都是255，255*25溢出，所以除保证了不溢出

    kernel_three = np.array([[0,-1,0],
                             [-1,5,-1],
                             [0,-1,0]],np.float32)
    # ddepth :-1 kernel:卷积核 dst：输出位置 anchor：卷积锚点 brodertype：边缘填充模式
    dst = cv.filter2D(image, -1, kernel=kernel_three)
    cv.imshow('custom_blur_demo', dst)

img = cv.imread('./WindowsLogo.jpg', 1)  # blue green red
blur_demo(img)
custom_blur_demo(img)
t1 = cv.getTickCount()

t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
