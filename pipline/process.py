import cv2 as cv
import numpy as np
import pipline.function as fc
import pipline.line as line
from matplotlib import pyplot as plt


def pipline(src, left_line, right_line, debug_flag):  # frame也代表了可以用图
    edge_img, resize_img, blur_resize_img = fc.auto_canny(src, debug_flag)
    lines = cv.HoughLines(edge_img, 1, np.pi / 180, 120)  # 霍夫检测

    key_points = fc.key_point_set(lines, resize_img, debug_flag)  # 更改成定时的才行

    point_x, point_y = fc.cal_key_point(key_points, resize_img, debug_flag)
    binary_final_roi, gray_fianl_roi, deal_image_roi, Mvt = fc.selec_roi(point_x, point_y, edge_img, blur_resize_img, 0,
                                                                         1)
    lane_binary_roi_img = fc.find_lane_key_color(gray_fianl_roi, deal_image_roi, binary_final_roi, debug_flag)
    # 检测车道线部分
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = fc.find_line_by_previous(lane_binary_roi_img,
                                                                                        left_line.current_fit,
                                                                                        right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = fc.find_line(lane_binary_roi_img, debug_flag)
    left_line.update(left_fit)
    right_line.update(right_fit)
    #lane_binary_roi_img 给的式合并后roi二值图像
    fc.draw_area(resize_img, lane_binary_roi_img,Mvt,left_fit,right_fit,point_y)



left_line = line.Line()
right_line = line.Line()

point_x = int(1280 / 2)
point_y = int(720 / 2)

img = cv.imread('../image/road_undist.jpg', 1)  # blue green red 三通道
t1 = cv.getTickCount()
pipline(img, left_line, right_line, 0)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
