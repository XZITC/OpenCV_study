import numpy as np
import cv2 as cv
from matplotlib import pyplot  as plt


# 图像缩小，resize和图像金字塔pryDown测试

def resize_demo(img):
    """
    resize法
    INTER_NN      -最近邻插值
    INTER_LINEAR  -双线性插值 (缺省使用)
    INTER_AREA    -使用象素关系重采样，当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 INTER_NN 方法。
    INTER_CUBIC   -立方插值。
    """
    # python中如果有个值是相等赋值，后面也都需要加上前缀
    imgGRB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h, w = img.shape[:2]  # 获取宽和高
    # 第二项记得转换下，宽 高
    dst = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)  # 法一 定倍数
    dst_1 = cv.resize(img, (int(w / 2), int(h / 2)), interpolation=cv.INTER_CUBIC)  # 法二  定宽高
    cv.imwrite('larger3.jpg', dst)
    print(dst.shape)


def pryDown_or_up_demo(image):
    #高斯金字塔，不行，比resize模糊好多
    #向下的图像金字塔用高斯模糊后抽取偶数行和列
    #向上的图像金字塔用拉普拉斯边缘加强后叠加
    r1 = cv.pyrDown(image)
    cv.imwrite('larger_jinzida.jpg', r1)


img = cv.imread('./image/high_way.jpg', 1)  # blue green red
#resize_demo(img)
t1 = cv.getTickCount()
pryDown_or_up_demo(img)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
