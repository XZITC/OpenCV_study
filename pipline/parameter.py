import numpy as np


class parameter():
    def __init__(self):
        # 消失点的坐标
        self.key_point_x = int(1280 / 2)
        self.key_point_y = int(720 / 2)
        #为True重新检测车道颜色 False为亮度没变化不用检测颜色
        self.brightness_changed = True
        # 车道颜色
        self.lane_color = [180, 180, 180]

    def update_lane_color(self, color):
        old_mean = np.mean(self.lane_color)
        new_mean = np.mean(color)
        if new_mean - old_mean < 100:
            self.lane_color = color

    def update_key_point(self, point_x, point_y):
        if int( point_x != 0) & int( point_y != 0):
            self.key_point_x = point_x
            self.key_point_y = point_y

    def judge_color_change(self,judge_num):
        if self.brightness_changed != judge_num:
            self.brightness_changed = judge_num
            return True
        else:
            return False