import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from class_9 import his_demo
from interval import Interval
import utils

line_color = np.ones([1, 1, 2], np.uint8)

pst_x = np.ones([1, 2], np.uint8)
lane_lag = Interval(0, 10)


def find_line_info(image, binary_roi_image):  # 返回的值是shaper过后的二值图
    h, w = image.shape[:2]
    lag = Interval(150, 255)
    for row in range(0, h, 1):
        for col in range(0, w - 1):
            if abs((int(binary_roi_image[row][col + 1]) - int(binary_roi_image[row][col]))) in lag:
                if len(pst_x) == 2:
                    int(pst_x[0][1]) - int(pst_x[0][0]) in lane_lag

                else:
                    pst_x[0][0] = col + 1
                    pst_x[0][1] = col + 1

                cv.circle(image, (col + 1, row), 1, (255, 255, 255), -1)
                # line_color = np.insert(line_color, row, values=image[row][col + 2], axis=0)
                cv.imshow('circle', image)


def deep_gray(image, c, b):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blank = np.zeros_like(gray)
    dst = cv.addWeighted(gray, c, blank, 1 - c, b)
    cv.imshow('con_bi_img', dst)
    return dst


def selec_roi(y, deal_image):  # ROI区域的绘制
    h, w = deal_image.shape[:2]  # 透视变换
    y = int(y + 20)
    pts11 = np.float32([[170, 720],  # 原图4点标注
                        [570, 450],
                        [700, 450],
                        [1100, 720]])

    pts22 = np.float32([[570, 720],  # 转换目标图4点标注
                        [570, 206],

                        [800, 206],
                        [800, 720]])
    M = cv.getPerspectiveTransform(pts11, pts22)
    Min = cv.getPerspectiveTransform(pts22,pts11)
    dst2 = cv.warpPerspective(deal_image, M, (1280, 720))  # 原图像的透视

    final_roi = dst2[y:deal_image.shape[0], :]
    return final_roi,Min


def perspective_transform(point_y,deal_image,info):
    dst = cv.warpPerspective(deal_image, info, (1280, 720))#写死了
    #dst = cv.warpPerspective(deal_image, info, (deal_image.shape[0], deal_image[1]))
    final_roi = dst[point_y:deal_image.shape[0], :]
    return final_roi

def get_perspective_info(deal_image):
    pts11 = np.float32([[170, 720],  # 原图4点标注
                        [570, 450],
                        [700, 450],
                        [1100, 720]])

    pts22 = np.float32([[570, 720],  # 转换目标图4点标注
                        [570, 206],
                        [800, 206],
                        [800, 720]])
    M = cv.getPerspectiveTransform(pts11, pts22)
    Min = cv.getPerspectiveTransform(pts22,pts11)
    return M, Min


def shaper(image):  # 锐化二值图像
    blur = cv.GaussianBlur(image, (3, 3), 0)
    kernel_three = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], np.float32)
    dst = cv.filter2D(image, -1, kernel=kernel_three)
    return dst


def threshold(gray):
    # 全局阈值二值化 主要用：cv.THRESH_OTSU和cv.THRESH_TRIANGLE两种
    # 这时p2为自动所以指定为0
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_TRIANGLE)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print(ret)
    cv.imshow('binary_otsu', binary)
    return binary


def find_lane_hsv(image, binary_roi_image, pts):  # 输入roi img彩色图 找到白色的rbg
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    load_colors_h = []
    load_colors_s = []
    load_colors_v = []

    load_colors_b = []
    load_colors_g = []
    load_colors_r = []
    load_colors = np.ones([1, 1, 3])
    h, w = binary_roi_image.shape[:2]
    for row in range(20, h, 3):
        for col in range(pts - 10, pts + 20):
            if (int(binary_roi_image[row][col]) - int(binary_roi_image[row][col - 1])) == 255:
                # 现在的方法，从给定的peak点从两侧像中点靠近，筛选出白色的点，白色使用rgb来筛选效果比较好
                # load_colors_h.append(hsv_img[row][col + 2][0])
                # load_colors_s.append(hsv_img[row][col + 2][1])
                # load_colors_v.append(hsv_img[row][col + 1][2])

                load_colors_b.append(image[row][col + 7][0])
                load_colors_g.append(image[row][col + 7][1])
                load_colors_r.append(image[row][col + 7][2])
                break
    for row in range(20, h, 3):
        for col in range(pts + 50, pts - 20, -1):
            if (int(binary_roi_image[row][col]) - int(binary_roi_image[row][col - 1])) == -255:
                # load_colors_h.append(hsv_img[row][col-1][0])
                # load_colors_s.append(hsv_img[row][col-1][1])
                # load_colors_v.append(hsv_img[row][col-1][2])

                load_colors_b.append(image[row][col - 8][0])
                load_colors_g.append(image[row][col - 8][1])
                load_colors_r.append(image[row][col - 8][2])
                break

    # h = int(np.mean(load_colors_h))
    # s = int(np.mean(load_colors_s))
    # v = int(np.mean(load_colors_v))
    # load_color = [h,s,v]

    b = int(np.mean(load_colors_b))
    g = int(np.mean(load_colors_g))
    r = int(np.mean(load_colors_r))
    load_color = [b, g, r]

    # counts_h = np.bincount(load_colors_h)
    # counts_s = np.bincount(load_colors_s)
    # counts_v= np.bincount(load_colors_v)
    #
    # # 返回众数
    # z_h = np.argmax(counts_h)
    # z_s = np.argmax(counts_s)
    # z_v = np.argmax(counts_v)
    # load_color_z = [z_h,z_s,z_v]
    return load_color


def extra_object_demo(image,roi_img, color):  # 提取颜色
    #hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hls = cv.cvtColor(roi_img,cv.COLOR_BGR2HLS)
    lower_1 = np.array(color) - 10  #白色rgb最低范围
    higher_1 = np.array(color) + 100  #白色rgb最高范围
    dst_w = cv.inRange(roi_img, lower_1, higher_1)  #提取白色rgb
    cv.imshow('hsvInRange', dst_w)
    yellower = np.array([10, 0, 90])  #黄色hls低范围
    yelupper = np.array([50, 255, 255])  #黄色hls高范围
    dst_y = cv.inRange(hls,  yellower, yelupper)
    cv.imshow('yellow_hls', dst_y)
    white_yellow_line = cv.bitwise_or(dst_w,dst_y,mask=None)
    cv.imshow('dsdsdsdssd',white_yellow_line)
    return white_yellow_line


img = cv.imread('../image/road_undist.jpg', 1)  # blue green red 三通道
t1 = cv.getTickCount() #计时开始


blur = cv.GaussianBlur(img, (3, 3), 5)
roi,Min = selec_roi(200, blur)
edge = cv.Canny(blur, 50, 150)
edge_roi,Min = selec_roi(200,edge)

deep_gray_image = deep_gray(blur, 1, 0) #加对比之后的灰度图

roi_img,Min = selec_roi(200, deep_gray_image)
grad_x = cv.Sobel(roi_img, cv.CV_32F, 1, 0)  # 在x方向上求梯度
gradx = cv.convertScaleAbs(grad_x)

dst = shaper(gradx) #对x方向的梯度灰度图像进行锐化
warped = threshold(dst) #对于得到图像二值化，用于后续提取白色信息

histogram = np.sum(dst, axis=0)
plt.plot(histogram)

midpoint = np.int(histogram.shape[0] / 2)
leftx_base = np.argmax(histogram[:midpoint])  # 寻找中心线左右附近的直方图peaks
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
#找白色的线的数值
lane_colors = find_lane_hsv(roi, warped, rightx_base)
#提取颜色车道线
color_load = extra_object_demo(img,roi ,lane_colors)

binary_warped = cv.bitwise_or(edge_roi,color_load,mask=None)
cv.imshow('binary',binary_warped)
# 开始拟合
#out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255


left_fit, right_fit, left_lane_inds, right_lane_inds,out_img = utils.find_line(binary_warped)
left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(binary_warped,left_fit,right_fit)
utils.draw_area(blur, binary_warped, Min, left_fit, right_fit)
# Choose the number of sliding windows

nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])


# left_fitx = left_fit[0]*ploty**3 + left_fit[1]*ploty**2 + left_fit[2]*ploty+left_fit[3]
# right_fitx = right_fit[0]*ploty**3 + right_fit[1]*ploty**2 + right_fit[2]*ploty+right_fit[3]
# left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
# right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
# plt.imshow(out_img)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)

# cv.imshow('roi', gradx)
# cv.imshow('warp_binary', roi_img)
# cv.imshow('canny', edge)

t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()


capture = cv.VideoCapture('../video/project_video.mp4')

while (False):
    ret, frame = capture.read()
    blur = cv.GaussianBlur(frame, (3, 3), 5)
    edge = cv.Canny(blur, 50, 150)
    deep_gray_image = deep_gray(blur, 1, 0)
    #roi_img = selec_roi(200, deep_gray_image)
    grad_x = cv.Sobel(roi_img, cv.CV_32F, 1, 0)  # 在x方向上求梯度
    gradx = cv.convertScaleAbs(grad_x)
    dst = shaper(gradx)
    cv.imshow('filter', dst)
    cv.imshow('edge', gradx)
    if cv.waitKey(60) == 32:  # 32 ascii空格
        capture.release()
        cv.destroyAllWindows()
        break
