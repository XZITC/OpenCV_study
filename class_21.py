import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


#  对象测量
def canny_demo(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)  # 高斯模糊去噪声
    edge_canny_one = cv.Canny(blur, 45, 150)    #  小波边缘检测？？？
    return edge_canny_one

#绘制外接框，计算面积 实时性不错
def measure_object(image):
    # canny边缘检测
    dst = canny_demo(image)
    contours, heriachy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #all_contours = cv.drawContours(image, contours, -1, (0, 0, 255))
    #cv.imshow('dst', all_contours)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)  # 计算每个轮廓的面积
        if area > 0:
            # 返回外接矩形的坐标和宽，高 （x，y）为矩形左上角的坐标，（w，h）是矩形的宽和高。
            x, y, w, h = cv.boundingRect(contour)
            mm = cv.moments(contour)  # 几何距
            type(mm)  # 字典类型
            cx = mm['m10'] / mm['m00']  # x重心坐标
            cy = mm['m01'] / mm['m00']  # y重心坐标
            #rate = mi 可以通过计算上下来看物体的粗细，瘦壮
            cv.circle(image, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)  # 绘制重心
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))  # 绘制矩
            #多边形逼近
            #approx_curve = cv.approxPolyDP(contour,4,True)
            #if approx_curve.shape[0] > 0 & approx_curve.shape[0]<4:
                #cv.drawContours(img,contours,i, (0, 0, 255))
            #print(approx_curve.shape)
    cv.imshow('measure_object', image)
    return image

def min_area_rect_demo(image): #带角度的最小外接矩形
    dst = canny_demo(image)
    contours, heriachy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)  # 计算每个轮廓的面积
        if area > 50:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(image,[box],0,(0,255,0))
    #cv.imshow('inAreaRect', image)
    return image,dst

img = cv.imread('./image/load_sunny.jpg', 1)  # blue green red
t1 = cv.getTickCount()
#measure_object(img)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()

capture = cv.VideoCapture(1)
#capture.set(3, 1280)
#capture.set(4,720)
while (True):
    ret, frame = capture.read()
    # frame = cv.flip(frame, -1)  # cv.filp()镜像变换 1：左右 -1上下
    #dst = canny_demo(frame)
    img,dst = min_area_rect_demo(frame)
    cv.imshow('hsvInRange', frame)
  #  cv.imshow('rawVideo', dst)
    # get_image_info(frame)
    if cv.waitKey(60) == 32:  # 32 ascii空格
        capture.release()
        cv.destroyAllWindows()
        break
