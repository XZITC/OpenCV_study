import numpy as np
import cv2 as cv
from matplotlib import pyplot  as plt

def lane_detection(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    h,w = img.shape[:2]
    roi = img[int(h/3)*2:h,0:w]
    #cv.imshow('roi',roi)
    dst = cv.Canny(roi,50,150,apertureSize=3)
    #cv.imshow('dst',dst)
    # houghLine输出的是r，theta图 houghLinesP输出直线
    # houghLine p2:r的长度，最长是1 p3：每次偏转的角度 np.pi/180 -> 1°
    lines = cv.HoughLines(dst,1,np.pi/180,50) #返回的是lines数组
    print(lines)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = rho*a
        y0 = rho*b
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow('hough_line',img)


def hough_lineP_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    roi = img[int(h / 3) * 2:h, 0:w]
    # cv.imshow('roi',roi)
    dst = cv.Canny(roi, 50, 150, apertureSize=3)
    cv.imshow('dst', dst)
    minLineLength = 50
    maxLineGap = 10
    lines = cv.HoughLinesP(dst, 1, np.pi / 180, 30, minLineLength, maxLineGap)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow('hough_lineP', img)


img = cv.imread('./image/load_sunny.jpg', 1)  # blue green red
t1 = cv.getTickCount()
#lane_detection(img)
hough_lineP_demo(img)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()