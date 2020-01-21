import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from class_9 import his_demo

def plt_draw_line(r, theta):  # 在plt中画线，不过好像有问题，画出来坐标系不对头
    x = np.linspace(0, 2048)
    a = np.cos(theta)
    b = np.sin(theta)
    y = (r - x * a) / b
    plt.xlim((0, 2048))
    plt.ylim((0, 1154))
    plt.plot(x, y, 'r')  # 红色，线宽为1个pixel
    plt.show()


def key_point_set(lines, image):
    # 每次传入两条hough直线的lines,取r和theta通过numpy计算直线的交点
    key_point = np.zeros([1, 2], np.float32)
    row = 0
    h, w = image.shape[:2]
    h_low = int(h * 0.2)  # 交点范围最低范围 画面1/5处
    h_high = int(h * 0.8)  # 交点最高范围 画面4/5处
    w_low = int(w * 0.2)
    w_high = int(h * 0.8)
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
                        cv.circle(image, (np.int(x[0][0]), np.int(x[1][0])), 5, (0, 255, 255), -1)
                        key_point = np.insert(key_point, row, values=x.T, axis=0)
                        row = row + 1
                        print(x)
                        cv.imshow('test', image)
                draw_two_line(line, image)
    point_x, point_y = cal_key_point(key_point, image)
    return point_x, point_y  # 返回均值化后的交点


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


def cal_key_point(set, image):  # 计算所有交点的均值
    if set[0][0] != 0.0:
        A = np.ones([1, set.shape[0]], np.float32)
        dot = A.dot(set)
        global point_x
        global point_y
        point_x = int(dot[0][0] / (set.shape[0] - 1))
        point_y = int(dot[0][1] / (set.shape[0] - 1))
        if (point_x < 4096) & (point_x > 0) & (point_y < 2160) & (point_y > 0):
            # 绘制点以及ROI区域的上线
            cv.circle(image, (point_x, point_y), 8, (255, 255, 255), -1)
            cv.line(image, (0, point_y), (image.shape[1], point_y), (0, 191, 255), 3)
    # cv.imshow('key point',image)
    return point_x, point_y  # 返回交点 point_y作用是划定roi的y轴的范围，不把天空划定进去


def selec_roi(x, y, binary_image, deal_image):  # ROI区域的绘制
    h, w = binary_image.shape[:2]  # 透视变换
    y = int(y + 20)
    pts1 = np.float32([[0, h],  # 原图4点标注
                       [int(2 * w / 5) + 100, y],
                       [int(3 * w / 5) - 50, y],
                       [w, h]])
    pts11 = np.float32([[170, 720],  # 原图4点标注
                        [570, 450],
                        [700, 450],
                        [1100, 720]])
    cv.circle(deal_image, (0, h), 8, (255, 255, 0), -1)
    cv.circle(deal_image, (int(2 * w / 5) + 50, y), 8, (255, 255, 0), -1)
    cv.circle(deal_image, (int(3 * w / 5) - 50, y), 8, (255, 255, 0), -1)
    cv.circle(deal_image, (w, h), 8, (255, 255, 0), -1)

    cv.line(deal_image, (0, h), (int(2 * w / 5), y), (0, 191, 255), 1)
    cv.line(deal_image, (int(2 * w / 5), y), (int(3 * w / 5), y), (0, 191, 255), 1)
    cv.line(deal_image, (int(3 * w / 5), y), (w, h), (0, 191, 255), 1)
    cv.imshow('pts1', deal_image)
    pts2 = np.float32([[0, h],  # 转换目标图4点标注
                       [0, 0],
                       [w, 0],
                       [w, h]])
    pts_y = int(720 - (w / 5) * (12 / 7))
    pts22 = np.float32([[570, 720],  # 转换目标图4点标注
                        [570, 206],
                        [800, 206],
                        [800, 720]])
    M = cv.getPerspectiveTransform(pts11, pts22)
    dst = cv.warpPerspective(binary_image, M, (1280, 720)) #二值图像的透视
    dst2 = cv.warpPerspective(deal_image, M, (1280, 720)) #原图像的透视
    final_roi = dst[y:binary_image.shape[0], :]
    cv.imshow('warp_binary', final_roi)
    cv.imwrite('../roi-image.jpg', final_roi)
    cv.imshow('warp', dst2)


def hough_lineP_demo(img):
    h, w = img.shape[:2]
    roi = img[int(h / 3) * 2:h, 0:w]
    # cv.imshow('roi',roi)
    dst = cv.Canny(roi, 50, 150, apertureSize=3)
    cv.imshow('dst', dst)
    minLineLength = 50
    maxLineGap = 10
    lines = cv.HoughLinesP(dst, 1, np.pi / 180, 30, minLineLength, maxLineGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow('hough_lineP', img)


def resize_image(image):
    shape_size = (1280, 720)  # 这里像以后统一缩小到720p
    # shape_size = (0, 0)
    dst = cv.resize(image, shape_size, fx=1, fy=1, interpolation=cv.INTER_CUBIC)
    print(dst.shape)
    # cv.imwrite('resize.jpg', dst)
    return dst


def shaper(image):
    kernel = np.ones([5, 5], np.float32) / 25  # 最大情况每个点都是255，255*25溢出，所以除保证了不溢出
    kernel_three = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], np.float32)
    # ddepth :-1 kernel:卷积核 dst：输出位置 anchor：卷积锚点 brodertype：边缘填充模式
    dst = cv.filter2D(image, -1, kernel=kernel_three)
    return dst


def auto_canny(image, sigma=0.33):
    small_img = resize_image(image)
    blur = cv.GaussianBlur(small_img, (5, 5), 5)  # 高斯模糊去噪声
    # compute the median of the single channel pixel intensities
    v = np.median(blur)
    # 计算中值，自动阈值canny检测
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(blur, lower, upper)
    # 返回边缘图像
    # edge = cv.Canny(blur, 50, 150)原始版本
    return edged, small_img


def get_combined_white(image):
    """
    #---------------------
    #提取白色车道线 思路是 利用直方图来判定图像最高大致的亮度分布，从而才确定白色因该取的范围
    #地平线下半部分的高亮区域 来确定白色
    # 现在是用rgb直接提取白色，感觉效果不错
    #
    """
    # frame = cv.flip(frame, -1)  # cv.filp()镜像变换 1：左右 -1上下
    lower_1 = np.array([100,100, 100])
    #lower_2 = np.array([[156, 43, 46]])
    higher_1 = np.array([255, 255, 255])
    #higher_2 = np.array([180, 255, 255])
    # inRange 1.hsv图 2.低值 3.高值
    imgc = image.copy
    rs = cv.inRange(image, lower_1, higher_1)
    cv.imshow('hsvInRange', rs)


def lane(binary, image):
    # houghLine输出的是r，theta图 houghLinesP输出直线
    # houghLine p2:r的长度，最长是1 p3：每次偏转的角度 np.pi/180 -> 1°
    lines = cv.HoughLines(binary, 1, np.pi / 180, 200)  # 返回的是lines数组 120
    print(img.shape)
    # if lines is not None:
    #    for line in lines:
    #         rho, theta = line[0]
    #         plt_draw_line(rho,theta)
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = rho * a
    #         y0 = rho * b
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))
    #         #plt_draw_line_2(x0, y0, a/-b)
    #         cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    point_x, point_y = key_point_set(lines, image)
    return point_x, point_y
    # cv.imshow('hough_line', image)


def extra_object_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_1 = np.array([16, 30, 50])
    higher_1 = np.array([34, 255, 255])
    # inRange 1.hsv图 2.低值 3.高值
    rs = cv.inRange(hsv, lower_1, higher_1)
    cv.imshow('hsvInRange', rs)

def source_import(src):
    # step one 降噪
    edges, small_img = auto_canny(src)
    point_x, point_y = lane(edges, small_img)
    return point_x, point_y, edges, small_img



img = cv.imread('../image/road_undist.jpg', 1)  # blue green red 三通道

t1 = cv.getTickCount()
#edge, small_img = auto_canny(img)
#get_combined_white(img)
#cv.imshow('hls_comble', edge)
#his_demo(img)

# selec_roi(866,480,small_img)
# rest2 = acny.auto_canny_fun(img)

#point_x, point_y, edged, smallimg = source_import(img)
#selec_roi(point_x, point_y, edged,smallimg)


# plt_draw_line(216,1.7453293)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()

capture = cv.VideoCapture('../video/challenge_video.mp4')
# capture = cv.VideoCapture(1)
# capture.set(cv.CAP_PROP_FRAME_COUNT, 30)

point_x = int(1280 / 2)
point_y = int(720 / 2)

while (True):
    ret, frame = capture.read()
    # print(frame.shape)

    #point_x, point_y, edged, smallimg = source_import(frame)
    #selec_roi(point_x, point_y, edged, smallimg)
    #get_combined_white(frame)
    rest1, small_img = auto_canny(frame)
    extra_object_demo(frame)
    cv.imshow('edge', rest1)
    # cv.imshow('2', rest2)
    if cv.waitKey(60) == 32:  # 32 ascii空格
        capture.release()
        cv.destroyAllWindows()
        break
