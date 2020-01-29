import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def resize_image(image):
    shape_size = (1280, 720)  # 这里像以后统一缩小到720p
    # shape_size = (0, 0)
    dst = cv.resize(image, shape_size, fx=1, fy=1, interpolation=cv.INTER_AREA)
    print(dst.shape)
    # cv.imwrite('resize.jpg', dst)
    return dst

def auto_canny(image, debug_flag, sigma=0.33):
    small_img = resize_image(image)
    blur = cv.GaussianBlur(small_img, (5, 5), 5)  # 高斯模糊去噪声
    # blur = cv.edgePreservingFilter(img, sigma_s=100, sigma_r=0.1, flags=cv.RECURS_FILTER)
    # compute the median of the single channel pixel intensities
    v = np.median(blur)
    # 计算中值，自动阈值canny检测
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(blur, lower, upper)
    # 返回边缘图像
    # edge = cv.Canny(blur, 50, 150)原始版本
    if debug_flag == 1:
        cv.imshow("canny",edged)
    return edged, small_img, blur


def draw_two_line(lines, image):  # 每次绘制两条直线
    # print(img.shape)
    if len(lines) == 2:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = rho * a
            y0 = rho * b
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # plt_draw_line_2(x0, y0, a/-b)
            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)


def key_point_set(lines, image, debug_flag):
    # 每次传入两条hough直线的lines,取r和theta通过numpy计算直线的交点
    key_points = np.zeros([1, 2], np.float32)
    row = 0
    h, w = image.shape[:2]
    h_low = int(h * 0.2)  # 交点范围最低范围 画面1/5处
    h_high = int(h * 0.8)  # 交点最高范围 画面4/5处
    if lines is not None:
        # a = lines[0:1]
        for i in range(0, len(lines), 2):  # 每次取两条计算交点和绘制
            line = lines[i:i + 2]
            if len(line) == 2:
                A = np.array([[np.cos(line[0][0][1]), np.sin(line[0][0][1])],
                              [np.cos(line[1][0][1]), np.sin(line[1][0][1])]])
                b = np.array([[line[0][0][0]],
                              [line[1][0][0]]])
                if np.linalg.det(A) != 0:  # 方程组的行列式不等于，才有解
                    # 计算坐标
                    x = np.linalg.solve(A, b)
                    # circle图上画坐标 x[0][0]-> x, x[1][0]->y
                    if (x[0][0] < 4096) & (x[0][0] > 0) & (x[1][0] < h_high) & (x[1][0] > h_low):
                        key_points = np.insert(key_points, row, values=x.T, axis=0)
                        row = row + 1
                        #调试模式启动！
                        if debug_flag == 1:
                            cv.circle(image, (np.int(x[0][0]), np.int(x[1][0])), 5, (0, 255, 255), -1)
                            cv.imshow('test', image)
                            draw_two_line(line, image) #绘制两条线
                            print(x)
    return key_points #返回中心交点集


def cal_key_point(set, image, debug_flag):  # 计算所有交点的均值
    """
    :param set: 中心点的集合
    :param image: 图像 用来画点，调试用
    :return:
    """
    if set[0][0] != 0.0:
        A = np.ones([1, set.shape[0]], np.float32)
        dot = A.dot(set)
        global point_x
        global point_y
        point_x = int(dot[0][0] / (set.shape[0] - 1))
        point_y = int(dot[0][1] / (set.shape[0] - 1))
        if (point_x < 4096) & (point_x > 0) & (point_y < 2160) & (point_y > 0):
            # 在规定区域内才绘制线
            if debug_flag == 1:
                cv.circle(image, (point_x, point_y), 8, (255, 255, 255), -1) # 消失点
                cv.line(image, (0, point_y), (image.shape[1], point_y), (0, 191, 255), 3)#地平线
                cv.imshow('key point',image)
    return point_x, point_y  # 返回交点 point_y作用是划定roi的y轴的范围，不把天空划定进去


def selec_roi(x, y, binary_image, deal_image, advanced_mode, debug_flag):  # ROI区域的绘制
    """

    :param x:
    :param y:
    :param binary_image:
    :param deal_image:
    :param advanced_mode:
    :param debug_flag:
    :return:
    """

    h, w = binary_image.shape[:2]  # 透视变换
    debug_image = deal_image
    gray = deep_gray(deal_image, 1, 0)
    if advanced_mode == 1:
        pts11 = np.float32([[170, 720],
                            [x-30, y+20],
                            [x+30, y+20],
                            [1100, 720]])

        pts22 = np.float32([[570, 720],
                            [570, 206],
                            [800, 206],
                            [800, 720]])
        M = cv.getPerspectiveTransform(pts11, pts22)
        Mvt = cv.getPerspectiveTransform(pts22, pts11)
    else:
        pts11 = np.float32([[170, 720],  # 原图4点标注
                            [570, 450],
                            [700, 450],
                            [1100, 720]])

        pts22 = np.float32([[570, 720],  # 转换目标图4点标注
                            [570, 206],
                            [800, 206],
                            [800, 720]])
        M = cv.getPerspectiveTransform(pts11, pts22)
        Mvt = cv.getPerspectiveTransform(pts22, pts11)
    dst = cv.warpPerspective(binary_image, M, (w, h))  # 二值图像的透视
    dst2 = cv.warpPerspective(gray, M, (w, h))  # 原图像的透视
    dst3 = cv.warpPerspective(deal_image, M, (w, h))
    binary_final_roi = dst[y:binary_image.shape[0], :]
    deal_image_roi = dst3[y:deal_image.shape[0], :]
    gray_fianl_roi = dst2[y:gray.shape[0], :]
    if debug_flag == 1:
        cv.circle(debug_image, tuple(pts11[0]), 8, (255, 255, 0), -1)
        cv.circle(debug_image, tuple(pts11[1]), 8, (255, 255, 0), -1)
        cv.circle(debug_image, tuple(pts11[2]), 8, (255, 255, 0), -1)
        cv.circle(debug_image, tuple(pts11[3]), 8, (255, 255, 0), -1)

        cv.line(debug_image, tuple(pts11[0]), tuple(pts11[1]), (0, 191, 255), 1)
        cv.line(debug_image, tuple(pts11[1]), tuple(pts11[2]), (0, 191, 255), 1)
        cv.line(debug_image, tuple(pts11[2]), tuple(pts11[3]), (0, 191, 255), 1)
        cv.imshow('selected_roi_image', binary_final_roi)
        cv.imshow('warp_binary', gray_fianl_roi)
        cv.imshow('warp', debug_image)
        #cv.imwrite('../roi-image.jpg', deal_fianl_roi)
    return binary_final_roi,gray_fianl_roi,deal_image_roi

def deep_gray(image, c, b):
    """
    增加对比度和亮度
    :param image:
    :param c:
    :param b:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blank = np.zeros_like(gray)
    dst = cv.addWeighted(gray, c, blank, 1 - c, b)
    return dst


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
    return binary

def find_lane_key_color(warp_roi_gray_image, warp_roi_image, warp_roi_binary_image,debuge_flag):
    grad_x = cv.Sobel(warp_roi_gray_image, cv.CV_32F, 1, 0)  # 在x方向上求梯度
    gradx = cv.convertScaleAbs(grad_x)
    shaperd = shaper(gradx)  # 对x方向的梯度灰度图像进行锐化
    binary_image = threshold(shaperd)  # 对于得到图像二值化，用于后续提取白色信息

    #通过直方图找车道中点
    histogram = np.sum(binary_image, axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])  # 寻找中心线左右附近的直方图peaks
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    lane_color = find_single_lane_color(warp_roi_image,binary_image,rightx_base)
    # find_single_lane_color(image, binary_image, leftx_base)
    lane_binary_roi_img = extra_object_demo(warp_roi_image,lane_color,debuge_flag)
    # lane_binary_roi_img = extra_object_demo(warp_roi_gray_image, lane_color, debuge_flag)
    binary_warped = cv.bitwise_or(warp_roi_binary_image, lane_binary_roi_img, mask=None)

    if debuge_flag == 1:
        # plt.plot(histogram)
        cv.imshow('Sobel', gradx)
        cv.imshow('threshold', binary_image)
        cv.imshow('binary_warped', binary_warped)

    return binary_warped

def find_single_lane_color(warp_roi_image, binary_roi_image, postion):  # 输入roi img彩色图 找到白色的rbg
    load_colors_b = []
    load_colors_g = []
    load_colors_r = []
    load_colors = np.ones([1, 1, 3])
    h, w = binary_roi_image.shape[:2]
    for row in range(20, h, 3):
        for col in range(postion - 10, postion + 20):
            if (int(binary_roi_image[row][col]) - int(binary_roi_image[row][col - 1])) == 255:
                # 现在的方法，从给定的peak点从两侧像中点靠近，筛选出白色的点，白色使用rgb来筛选效果比较好
                load_colors_b.append(warp_roi_image[row][col + 7][0])
                load_colors_g.append(warp_roi_image[row][col + 7][1])
                load_colors_r.append(warp_roi_image[row][col + 7][2])
                break
    for row in range(20, h, 3):
        for col in range(postion + 50, postion - 20, -1):
            if (int(binary_roi_image[row][col]) - int(binary_roi_image[row][col - 1])) == -255:
                load_colors_b.append(warp_roi_image[row][col - 8][0])
                load_colors_g.append(warp_roi_image[row][col - 8][1])
                load_colors_r.append(warp_roi_image[row][col - 8][2])
                break

    b = int(np.mean(load_colors_b))
    g = int(np.mean(load_colors_g))
    r = int(np.mean(load_colors_r))
    load_color = [b, g, r]

    return load_color


def extra_object_demo(roi_img, color, debuge_flag):  # 提取颜色 明天考虑下灰度图提取
    #hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #白色
    lower_1 = np.array(color) - 30  #白色rgb最低范围
    higher_1 = np.array(color) + 100  #白色rgb最高范围
    dst_w = cv.inRange(roi_img, lower_1, higher_1)  #提取白色rgb

   #黄色
    hls = cv.cvtColor(roi_img,cv.COLOR_BGR2HLS)
    yellower = np.array([10, 0, 90])  #黄色hls低范围
    yelupper = np.array([50, 255, 255])  #黄色hls高范围
    dst_y = cv.inRange(hls,  yellower, yelupper)

    #两个色彩二值并一下输出
    white_yellow_line = cv.bitwise_or(dst_w,dst_y,mask=None)

    if debuge_flag == 1:
        cv.imshow('hsvInRange', dst_w)
        cv.imshow('yellow_hls', dst_y)
        cv.imshow('dsdsdsdssd', white_yellow_line)

    return white_yellow_line


def find_line(binary_warped,debuge_flag):  # binary_warped是roi区域的二值图 功能：滑窗寻找两侧道路
    # 测试的输出图
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped, axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
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
        #cv.imshow('img',out_img)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 3)
    if debuge_flag == 1:
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
        right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
    return left_fit, right_fit, left_lane_inds, right_lane_inds,out_img
    #这个返回的的可以画框


def find_line_by_previous(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 3) + left_fit[1] * (nonzeroy ** 2) +
    left_fit[2] * nonzeroy + left_fit[3] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 3) +
    left_fit[1] * (nonzeroy ** 2) + left_fit[2] * nonzeroy + left_fit[3] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 3) + right_fit[1] * (nonzeroy ** 2) +
    right_fit[2] * nonzeroy + right_fit[3] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 3) +
    right_fit[1] * (nonzeroy ** 2) + right_fit[2] * nonzeroy + right_fit[3] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 3)
    return left_fit, right_fit, left_lane_inds, right_lane_inds
    #这个但返回是画线


def draw_area(undist, binary_warped, Minv, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # 3次拟合
    left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
    right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    undist = np.array(undist)
    newwarp = cv.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    #newwarp = expand(newwarp)
    # Combine the result with the original image
    result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)
    cv.imshow('res',result)
    cv.imshow('warp', newwarp)
    return result, newwarp

def expand(img): #对区域进行填充
    image = img
    _,green,_ = cv.split(image)
    s = np.sum(green,axis=1)
    a = range(720)
    for i in reversed(a):
        if s[i] <200:
            break
        for j in range(1280):  # min x
            if green[i][j] == 255:
                break
        for k in reversed(range(1280)): #max x
            if green[i][k] == 255:
                break
        for l in range(int(s[i]/255)): # s[i]/255  the number
            image[i,j-l,2] = 255
        for l in range(int(s[i]/255)):
            image[i,k+l,2] = 255
    return image