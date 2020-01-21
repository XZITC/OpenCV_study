import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from class_9 import his_demo
from interval import Interval

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
    dst2 = cv.warpPerspective(deal_image, M, (1280, 720))  # 原图像的透视
    final_roi = dst2[y:deal_image.shape[0], :]
    return final_roi


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


def find_lane_hsv(image, binary_roi_image, pts):  # 输入roi img彩色图
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
    lower_1 = np.array(color) - 10 #白色rgb最低范围
    higher_1 = np.array(color) + 100 #白色rgb最高范围
    dst_w = cv.inRange(roi_img, lower_1, higher_1) #提取白色rgb
    cv.imshow('hsvInRange', dst_w)
    yellower = np.array([10, 0, 90]) #黄色hls低范围
    yelupper = np.array([50, 255, 255]) #黄色hls高范围
    dst_y = cv.inRange(hls,  yellower, yelupper)
    cv.imshow('yellow_hls', dst_y)
    white_yellow_line = cv.bitwise_or(dst_w,dst_y,mask=None)
    cv.imshow('dsdsdsdssd',white_yellow_line)
    return white_yellow_line


img = cv.imread('../image/road_undist.jpg', 1)  # blue green red 三通道
t1 = cv.getTickCount()

print(img.shape)
blur = cv.GaussianBlur(img, (3, 3), 5)
edge = cv.Canny(blur, 50, 150)
edge_roi = selec_roi(200,edge)

deep_gray_image = deep_gray(blur, 1, 0)
roi = selec_roi(200, blur)
roi_img = selec_roi(200, deep_gray_image)
grad_x = cv.Sobel(roi_img, cv.CV_32F, 1, 0)  # 在x方向上求梯度
gradx = cv.convertScaleAbs(grad_x)

dst = shaper(gradx)
warped = threshold(dst)

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
out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(dst.shape[0] / nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window + 1) * window_height
    win_y_high = binary_warped.shape[0] - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
# 拟合结束

# find_line_info(roi_img, dst)
cv.imshow('roi', gradx)
cv.imshow('warp_binary', roi_img)
cv.imshow('canny', edge)

t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()

capture = cv.VideoCapture('../video/project_video.mp4')
# capture = cv.VideoCapture(1)
# capture.set(cv.CAP_PROP_FRAME_COUNT, 30)


while (False):
    ret, frame = capture.read()
    blur = cv.GaussianBlur(frame, (3, 3), 5)
    edge = cv.Canny(blur, 50, 150)
    deep_gray_image = deep_gray(blur, 1, 0)
    roi_img = selec_roi(200, deep_gray_image)
    grad_x = cv.Sobel(roi_img, cv.CV_32F, 1, 0)  # 在x方向上求梯度
    gradx = cv.convertScaleAbs(grad_x)
    dst = shaper(gradx)
    cv.imshow('filter', dst)
    cv.imshow('edge', gradx)
    # cv.imshow('2', rest2)
    if cv.waitKey(60) == 32:  # 32 ascii空格
        capture.release()
        cv.destroyAllWindows()
        break
