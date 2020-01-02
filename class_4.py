import numpy as np
import cv2 as cv

# 第四次课
from class_1 import img


def add_demo_complx(m1, m2):
    dst = np.zeros([240, 320, 3], np.uint8)
    for row in range(240):
        for col in range(320):
            for ch in range(3):
                res = int(m1[row, col, ch]) + int(m2[row, col, ch])
                if (res >= 255):
                    dst[row, col, ch] = 255
                else:
                    dst[row, col, ch] = res
    cv.imshow('dst', dst)


def add_demo(m1, m2):
    dst = cv.add(m1, m2)  # add函数，把两张图片相加，相同大小
    cv.imshow('dst', dst)


def subtra_demo(m1, m2):  # 减法
    dst = cv.subtract(m1, m2)
    cv.imshow('dst', dst)


def divide_demo(m1, m2):  # 除法
    dst = cv.divide(m1, m2)
    cv.imshow('dst', dst)


def multiply_demo(m1, m2):  # 乘法 att：图像带平滑，边缘有锯齿导致相乘后边缘是很多杂色
    dst = cv.multiply(m1, m2)
    cv.imshow('dst', dst)


# mean求各通道平均值
def mean_demo(m1, m2):
    M1 = cv.mean(m1)
    M2 = cv.mean(m2)
    MF_1 = cv.meanStdDev(m1)  # 各通道标准差 标准差越大，图像contrast越强烈 纯色图像标准差为0
    MF_2 = cv.meanStdDev(m2)  # 例如运用在扫描仪上，判断图像是否为空内容
    return M1, M2, MF_1, MF_2


# 调整亮度，对比度
def adjust_contrast_bright(img, c, b):
    # dst = src1[I]*alpha + src2[I]*beta + gamma (cv.addsWeights)
    # 加权,本质上原理是每个像素乘上C,把各数值的扩大各像素之间的差距从而影响对比度最高255
    # 亮度，0-255越高越亮，所以整体加上一个值来提高亮度，最高255。这里的blank是为了补参数实则无用
    # 自己实现：cv.mutiply() + cv.adds() 但效率没有直接用addsWeight来的快
    # cv.addsWeights支持小数，是带取整的小数，自己实现还需要考虑小数以及类型的转换
    blank = np.zeros_like(img)
    dst = cv.addWeighted(img, c, blank, 1 - c, b)
    cv.imshow('con_bi_img', dst)
    # cv.imshow('con_bi_img2', dst2)


def logic_cal_demo(img1, img2):
    dst_and = cv.bitwise_and(img1, img2)  # 与运算 输出共同的不为零的区域
    dst_or = cv.bitwise_or(img1, img2)  # 或运算 输出两者任意非零的区域
    dst_not = cv.bitwise_not(img1)  # 反色
    cv.imshow('dst_and', dst_and)
    cv.imshow('dst_or', dst_or)
    cv.imshow('dst_or', dst_not)


img_1 = cv.imread('C:/Project/PycharmProjects/OpenCV_study/LinuxLogo.jpg', 1)  # blue green red 三通道顺序
img_2 = cv.imread('C:/Project/PycharmProjects/OpenCV_study/WindowsLogo.jpg', 1)
# cv.imshow('img_1', img_1)
# cv.imshow('img_2', img_2)
t1 = cv.getTickCount()  # 获取当前cpu周期时间
# logic_cal_demo(img_1,img_2)
# m1,m2,mf_1,mf_2 = mean_demo(img_1, img_2)
# print(m1,m2,mf_1,mf_2)
adjust_contrast_bright(img_2, 1.5, 2)
t2 = cv.getTickCount()  # 同获取时间
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
