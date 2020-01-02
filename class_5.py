# ROI 泛洪填充
import cv2 as cv
import numpy as np


# 简单roi区 变换再塞回去
def ROI(src, row1, row2, col1, col2):
    roi = src[row1:row2, col1:col2]  # roi区域选取
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)  # 变换到gray，三维转二维
    verColor = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # 灰度图转彩色，本质是有二维数组转三维数组
    src[row1:row2, col1:col2] = verColor
    cv.imshow('ROI', src)
    return roi


# 泛洪填充 BGR
def flood_color(image):
    copyImage = image.copy()  # .copy自带图像复制
    h, w = image.shape[:2]  # 获取输入图像宽，长
    mask = np.zeros([h + 2, w + 2], np.uint8)  # 生成填充遮罩 att：长+2，宽+2 必须 类型为uint8
    """
    p1：填充图像 p2：遮罩 p3：发散点 
    p4：loDiff 当前观察像素值与其部件邻域像素值或待加入该部件的种子像素之间的亮度或颜色之负差（lower brightness/color difference）的最大值
            src-loDuff 各通道值相减<=src这点像素值<=src+upDuff 各通道值相加
    p5:upDiff 当前观察像素值与其部件邻域像素值或待加入该部件的种子像素之间的亮度或颜色之正差（lower brightness/color difference）的最大值
    p6:flag FLOODFILL_FIXED_RANGE 彩色图像 
    """
    cv.floodFill(copyImage, mask, (30, 30), (0, 255, 255), (255, 255, 255), (0, 0, 0), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow('flood', copyImage)


# 二值图像的填充
def flood_binary():
    img = np.zeros([400, 400, 3], np.uint8)  # 创建一个黑色的二值图像
    img[100:300, 100:300, :] = 255  # 对选定roi区域进行图像填充
    """
    1.二值图像创建mask有特别要求，mask： *初始化为1 *坐标需+2 *单通道 *uint8
    2.把mask中对应的ROI区域填充为0 
    3.cv.FLOODFILL_MASK_ONLY二值填充
    """
    mask = np.ones([402, 402, 1], np.uint8)
    mask[101:301, 101:301] = 0
    cv.floodFill(img, mask, (200, 200), (0, 255, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow('roi', img)


img = cv.imread('C:/Project/PycharmProjects/OpenCV_study/WindowsLogo.jpg', 1)  # blue green red
t1 = cv.getTickCount()
# ROI(img, 20, 200, 20, 200)
flood_binary()
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
