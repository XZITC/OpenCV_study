import numpy as np
import cv2 as cv


# 二值化图像

def threshold_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 全局阈值二值化 主要用：cv.THRESH_OTSU和cv.THRESH_TRIANGLE两种
    # 这时p2为自动所以指定为0
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_TRIANGLE)
    print(ret)
    cv.imshow('binary', binary)


def adapt_threshold_demo(img):
    # 自适应的局部阈值二值化
    # img_blur = cv.medianBlur(img, 3)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.ADAPTIVE_THRESH_GAUSSIAN_C ,cv.ADAPTIVE_THRESH_MEAN_C两者二值计算法
    # p4：区域选择大小，必须奇书
    # p5：好比偏置，激活值

    dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 25, 10)
    cv.imwrite('全局.jpg',dst)
    cv.imshow('binary', dst)


def video_adapt_threshold_demo():
    capture = cv.VideoCapture(0)
    while (True):
        ret, frame = capture.read()
        # frame = cv.flip(frame, -1)  # cv.filp()镜像变换 1：左右 -1上下
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 25, 10)
        # res = cv.bilateralFilter(dst, 0, 100, 5) 边缘保留速度慢
        cv.imshow('hsvInRange', dst)
        cv.imshow('rawVideo', frame)
        # get_image_info(frame)
        if cv.waitKey(60) == 32:  # 32 ascii空格
            capture.release()
            cv.destroyAllWindows()
            break


img = cv.imread('./image/load_sunny.jpg', 1)  # blue green red
#threshold_demo(img)
#video_adapt_threshold_demo()
t1 = cv.getTickCount()
adapt_threshold_demo(img)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
