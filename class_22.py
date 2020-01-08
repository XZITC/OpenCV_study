import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#二值图像膨胀腐蚀
capture = cv.VideoCapture(1)
capture.set(3, 1280)
capture.set(4,720)

while (True):
    ret, frame = capture.read()
    print(frame.shape)
    #frame = cv.flip(frame, -1)  # cv.filp()镜像变换 1：左右 -1上下
    cv.namedWindow('video', cv.WINDOW_NORMAL)
    cv.imshow('video', frame)

    #cv.imshow('rawVideo', dst)
    # get_image_info(frame)
    if cv.waitKey(60) == 32:  # 32 ascii空格
        capture.release()
        cv.destroyAllWindows()
        break