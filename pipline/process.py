import cv2 as cv
import numpy as np
import pipline.function as fc
import pipline.line as line
import pipline.parameter as par
from matplotlib import pyplot as plt
from PIL import Image


def pipline(src, left_line, right_line, parameter, frame_num, debug_flag):  # frame也代表了可以用图
    edge_img, resize_img, blur_resize_img = fc.auto_canny(src, debug_flag)
    # 第一帧初始化，需要检测消失点，检测车道线颜色,或者每两秒检测一次
    if frame_num == 1 or frame_num % 48 == 0:
        lines = cv.HoughLines(edge_img, 1, np.pi / 180, 200)  # 霍夫检测
        key_points = fc.key_point_set(lines, resize_img, debug_flag)  # 更改成定时的才行
        if key_points[0][0] > 0:
            key_point_x, key_point_y = fc.cal_key_point(key_points, resize_img, debug_flag)
            parameter.update_key_point(key_point_x, key_point_y)
    # extra 待会用作HoughLineP
    binary_final_roi, binary_final_extra_roi, gray_fianl_roi, deal_image_roi, Mvt = fc.selec_roi(parameter.key_point_x,
                                                                                                 parameter.key_point_y,
                                                                                                 edge_img,
                                                                                                 blur_resize_img, 0,
                                                                                                 1)

    image_bright = fc.judge_brightness(resize_img, debug_flag)
    # 这边做一个颜色定时的功能
    if frame_num == 1 or parameter.judge_color_change(judge_num=image_bright) == 0:
        lane_color = fc.find_lane_key_color(gray_fianl_roi, deal_image_roi, binary_final_roi, frame_num, debug_flag)
        parameter.update_lane_color(lane_color)
        print(parameter.lane_color)
    lane_binary_roi_img = fc.extra_object_demo(deal_image_roi, parameter.lane_color, debug_flag)

    binary_warped = cv.bitwise_or(binary_final_roi, lane_binary_roi_img, mask=None)
    cv.imshow('binary_warped',binary_warped)
    cv.imshow('binary_final_extra_roi', binary_final_extra_roi)
    # 检测车道线部分
    # if left_line.detected and right_line.detected:
    #     left_fit, right_fit, left_lane_inds, right_lane_inds = fc.find_line_by_previous(lane_binary_roi_img,
    #                                                                                     left_line.current_fit,
    #                                                                                     right_line.current_fit)
    # else:
    left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = fc.find_line(binary_warped, debug_flag)
    left_line.update(left_fit)
    right_line.update(right_fit)
    # lane_binary_roi_img 给的式合并后roi二值图像
    fc.draw_area(resize_img, lane_binary_roi_img, deal_image_roi, Mvt, left_fit, right_fit, parameter.key_point_y,
                 debug_flag)


para = par.parameter()
left_line = line.Line()
right_line = line.Line()

point_x = int(1280 / 2)
point_y = int(720 / 2)

img = cv.imread('../image/s1.jpg', 1)  # blue green red 三通道
# img_2 = Image.open('../image/s1.jpg')
# img_2 = img_2.resize((1280, 720),Image.BICUBIC)
t1 = cv.getTickCount()
# pipline(img, left_line, right_line, para, 0, 0)

t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()

capture = cv.VideoCapture('../video/C0034.mp4')
capture.set(cv.CAP_PROP_POS_FRAMES,143)
point_x = int(1280 / 2)
point_y = int(720 / 2)

debug = 0
while (True):
    ret, frame = capture.read()
    frame_num = int(capture.get(cv.CAP_PROP_POS_FRAMES))
    # pipline(img, left_line, right_line, frame_num, 0)
    pipline(frame, left_line, right_line, para, frame_num, debug)
    if cv.waitKey(60) == 32:  # 32 ascii空格
        capture.release()
        cv.destroyAllWindows()
        break
