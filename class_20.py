import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

 #不必用edge 求梯度后，求出梯度总和，用梯度自动阈值和二值化，从而避免求边缘的时候阈值带来的烦恼
 # 比求完梯度后求边缘得出结果要好