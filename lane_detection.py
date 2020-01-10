import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plt_draw_line(r, theta):  # 在plt中画线，不过好像有问题，画出来坐标系不对头
    x = np.linspace(0, 2048)
    a = np.cos(theta)
    b = np.sin(theta)
    y = (r - x * a) / b
    plt.xlim((0, 2048))
    plt.ylim((0, 1154))
    plt.plot(x, y, 'r')  # 红色，线宽为1个pixel
    plt.show()


def cal_key_point(lines, image):
    # 每次传入两条hough直线的lines,取r和theta通过numpy计算直线的交点
    point = np.zeros([lines.size, 2], np.float32)
    if len(lines) == 2:
        A = np.array([[np.cos(lines[0][0][1]), np.sin(lines[0][0][1])],
                      [np.cos(lines[1][0][1]), np.sin(lines[1][0][1])]])
        b = np.array([[lines[0][0][0]],
                      [lines[1][0][0]]])
        if np.linalg.det(A) != 0:  # 方程组的行列式不等于，才有解
            # 计算坐标
            x = np.linalg.solve(A, b)
            # circle图上画坐标
            if (x[0][0] < 4096) &(x[0][0] > 0) & (x[1][0]<2160) & (x[1][0] > 0):
                cv.circle(image, (np.int(x[0][0]), np.int(x[1][0])), 5, (0, 255, 255), -1)
    draw_two_line(lines, image)
    # if x is not None:


def draw_two_line(lines, image):
    print(img.shape)
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


def resize_func(image):
    dst = cv.resize(image, None, fx=0.4, fy=0.4, interpolation=cv.INTER_CUBIC)
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


def canny_demo(image):
    small_img = resize_func(image)
    blur = cv.GaussianBlur(small_img, (5, 5), 5)  # 高斯模糊去噪声
    # cv.imshow('blur', blur)
    edge_canny_one = cv.Canny(blur, 50, 150)
    cv.imshow('canny', edge_canny_one)
    return edge_canny_one, small_img


def lane(binary, image):
    # houghLine输出的是r，theta图 houghLinesP输出直线
    # houghLine p2:r的长度，最长是1 p3：每次偏转的角度 np.pi/180 -> 1°
    lines = cv.HoughLines(binary, 1, np.pi / 180, 120)  # 返回的是lines数组 120
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
    if lines is not None:
        # a = lines[0:1]
        for i in range(0, len(lines), 2):
            b = lines[i:i + 2]
            cal_key_point(b, image)
    return image
    # cv.imshow('hough_line', image)


def socre_import(src):
    # step one 降噪
    canny_res, small_img = canny_demo(src)
    res = lane(canny_res, small_img)
    return res


img = cv.imread('./image/lane_3.jpg', 1)  # blue green red
t1 = cv.getTickCount()
rest = socre_import(img)
cv.imshow('hsvInRange', rest)
# plt_draw_line(216,1.7453293)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()

#capture = cv.VideoCapture('./video/sample.flv')
capture = cv.VideoCapture(0)
#capture.set(cv.CAP_PROP_FRAME_COUNT, 30)
while (False):
    ret, frame = capture.read()
    # print(frame.shape)
    rest = socre_import(frame)
    cv.imshow('hsvInRange', rest)
    if cv.waitKey(60) == 32:  # 32 ascii空格
        capture.release()
        cv.destroyAllWindows()
        break
