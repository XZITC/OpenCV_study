import numpy as np
import cv2 as cv

# 图像分割二值化


def plot_threshold(frame, row, col):
    """
    分块二值化图像，速度稍有提升 1080p因该用不到
    :param frame:
    :param row: 划分为几行
    :param col: 划分为几列
    :return:
    """
    img_h, img_w = frame.shape[:2]
    step_row = int(img_h / row)
    step_col = int(img_w / col)
    h = step_row
    w = step_col
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #先灰度化
    for ro in range(0, img_h, step_row):
        for co in range(0, img_w, step_col):
            roi_frm = gray[ro:h + ro, co:w + co]
            dst = cv.adaptiveThreshold(roi_frm, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY, 25, 10)
            gray[ro:h + ro, co:w + co] = dst
    cv.imwrite('test2.jpg', gray)


def adapt_threshold_demo(img): #全局的二值化
    # 自适应的局部阈值二值化
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 25,10)
    #cv.imwrite('test1.jpg', dst)
    #cv.imshow('binary', dst)

img = cv.imread('./image/cam_test.jpg', 1)  # blue green red
# cv.setUseOptimized(True)
# cv.useOptimized()
t1 = cv.getTickCount()
plot_threshold(img, 5, 5)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
