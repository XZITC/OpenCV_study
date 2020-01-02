import numpy as np
import cv2 as cv

def color_space_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    cv.imshow('gray', gray)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow('hsv', hsv)


# 提通过hsv提取对应颜色
def extra_object_demo():
    capture = cv.VideoCapture(1)
    while (True):
        ret, frame = capture.read()
        # frame = cv.flip(frame, -1)  # cv.filp()镜像变换 1：左右 -1上下
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_1 = np.array([0, 43, 46])
        lower_2 = np.array([[156, 43, 46]])
        higher_1 = np.array([10, 255, 255])
        higher_2 = np.array([180, 255, 255])
        rs = cv.inRange(hsv, lower_1, higher_1)
        rs_0 = cv.inRange(hsv, lower_2, higher_2)
        mask = cv.add(rs, rs_0) #生成遮罩
        # bitwise_and 显示共同部分，达到提取mask区域图像目的
        dst = cv.bitwise_and(frame, frame, mask=mask)  # mask不能直接传参因为维度不同
        cv.imshow('hsvInRange', dst)
        cv.imshow('rawVideo', frame)
        # get_image_info(frame)
        if cv.waitKey(60) == 32:  # 32 ascii空格
            capture.release()
            cv.destroyAllWindows()
            break


extra_object_demo()

