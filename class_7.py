import numpy as np
import cv2 as cv

def gauss_blur(img):
    """
      把卷积核换成高斯核（简单来说，方框不变，将原来每个方框的值是相等的，
   现在里面的值是符合高斯分布的，方框中心的值最大，其余方框根据距离中心元素的距离递减，
   构成一个高斯小山包。原来的求平均数现在变成求加权平均数，全就是方框里的值）。
      实现的函数是 cv2.GaussianBlur()。我们需要指定高斯核的宽和高（必须是奇数）。
   以及高斯函数沿 X， Y 方向的标准差。如果我们只指定了 X 方向的的标准差， Y 方向也会取相同值。
   如果两个标准差都是 0，那么函数会根据核函数的大小自己计算。
   高斯滤波可以有效的从图像中去除高斯噪音。
    """
    dst = cv.GaussianBlur(img,(0,0),1)
    dst_1 = cv.GaussianBlur(img,(11,11),1)
    cv.imshow('gass', dst)
    cv.imshow('gass2', dst_1)
img = cv.imread('./WindowsLogo.jpg', 1)  # blue green red

t1 = cv.getTickCount()
gauss_blur(img)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()