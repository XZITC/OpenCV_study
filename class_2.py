import numpy as np
import cv2 as cv


def access_pixels(img):
    print(img.shape)
    hight = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    print('hight: %s,width: %s,channels: %s' % (hight, width, channels))
    for row in range(hight):
        for col in range(width):
            for ch in range(channels):
                pv = img[row, col, ch]
                img[row, col, ch] = 255 - pv
    cv.imshow('afEdit,img', img)


def create_img(): # 创建400*400黑色图像
    img = np.zeros([400, 400, 3], np.uint8)  # uint8 range 0-255
    cv.imshow('create_img', img)


# 改变通道数量 傻瓜式，效率低
def change_channels(img, ch):
    if (ch < 3):
        hight = img.shape[0]
        width = img.shape[1]
        for row in range(hight):
            for col in range(width):
                pv = img[row, col, ch]
                img[row, col, ch] = 255
        cv.imshow('create_img', img)
    else:
        print('out of except channels')


def change_channels_1(img, ch):
    hight = img.shape[0]
    width = img.shape[1]
    img[:, :, ch] = np.ones([hight, width]) * 255
    cv.imshow('create_img', img)


# 图片读取 读取注意转换，图片：宽x高 -> 矩阵：行x列 三位数组：三通道
img = cv.imread('C:/Users/XZITC/Desktop/2.jpg', 1)  # blue green red 三通道顺序
t1 = cv.getTickCount()  # 获取当前cpu周期时间
# change_channels(img, 0)
change_channels_1(img, 0)
t2 = cv.getTickCount()  # 同获取时间
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
# waitkey()很重要的函数，等待键盘输入事件时间 ord()获取AsCII码
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
