# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 22:45:09 2025

@author: 杨艺楠
"""

# constants.py - 定义常量 (对应 Fortran constants_mod)
import numpy as np

# 物理和数学常量
# 圆周率
pi = 3.14159265358979323846264338327950288419716939937510
deg2rad = pi / 180.0                              # 度到弧度转换因子
rad2deg = 1.0 / deg2rad                           # 弧度到度转换因子
rearth = 6.3712e6                                 # 地球半径 (米)
omega = 7.2921e-5                                # 地球自转角速度 (1/秒)

# 数值常量
one = 1.0
zero = 0.0

# 时间常量 (以秒计)
hour = 3600.0
day = 24.0 * hour

# 误差阈值和缺省值
delt = 1.0e-8                                    # 判断浮点数近似相等的阈值
undef = np.nan                                     # 未定义或缺失值的标记
