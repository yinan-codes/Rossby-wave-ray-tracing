# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 22:46:04 2025

@author: 杨艺楠
"""

# interpolation.py - 双线性插值模块
import numpy as np

from constants import delt  # , undef
# from metpy.interpolate import interpolate_to_points


def check_err(ier):
    """检查误差代码 ier，输出相应错误信息。"""
    if ier == 0:
        return
    elif ier == 1:
        print("not enough points in input/output array")
    else:
        print("xi or yi are not monotonically increasing")


def dmonoinc(arr):
    """检查数组 arr 是否严格单调递增。返回0表示单调递增，非0表示不满足要求。"""
    n = len(arr)
    if n < 2:
        return 1  # 数据点不足
    # 判断严格递增
    for i in range(1, n):
        if arr[i] <= arr[i - 1]:
            return 2
    return 0


def batch_linint2_metpy(xi, yi, fi, xo, yo, fo_missing=np.nan,
                        xcyclic=False, mode='original', all_dtype='float64'):
    """
    批量插值函数：对多个点 (xo, yo) 在网格 fi 上插值。
    参数:
        xi, yi      - 一维网格坐标
        fi          - 2D 数据，形状 (len(xi), len(yi))
        xo, yo      - 多个插值点，一维数组
        fo_missing  - 缺测值
        xcyclic     - 是否对 xi 做周期处理
    返回:
        values      - 插值后的值，形状 (len(xo),)
    """
    if mode == 'original':
        xi, yi = np.asarray(xi), np.asarray(yi)
        fi = np.asarray(fi)

        # x 方向周期处理
        if xcyclic:
            dx = xi[1] - xi[0]
            period = (xi[-1] - xi[0]) + dx
            xo = ((xo - xi[0]) % period) + xi[0]

            xi = np.concatenate(([xi[0] - dx], xi, [xi[-1] + dx]))
            fi = np.concatenate((fi[-1:, :], fi, fi[:1, :]), axis=0)

        # 构造插值点数组
        points = np.column_stack((yo, xo))  # shape: (N, 2)

        # 展开网格数据
        grid_x, grid_y = np.meshgrid(xi, yi, indexing='ij')
        grid_points = np.column_stack((grid_y.ravel(), grid_x.ravel()))
        grid_values = fi.ravel()

        mask = grid_values != fo_missing
        if not np.any(mask):
            return np.full_like(xo, fo_missing)

        return interpolate_to_points(
            grid_points[mask], grid_values[mask], points)
    elif mode == 'numpy':
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        lons = xo % (2 * np.pi)
        ilons = (lons - xi[0]) / dx
        ilats = (yo - yi[0]) / dy
        # print('ilon&ilat',ilons[0],ilats[0])
        return bilinear_interpolation_(
            fi[None, :, :, :], ilons, ilats, xcyclic=xcyclic, all_dtype=all_dtype)


'''
add at 20250407 22:34
'''


def get_pixel_value(imgs, x, y):
    '''
    imgs: met fields, shape=(1,lon,lat,vars)
    x: indices in lons
    y: indices in lats
    '''

    return imgs[0, x, y, :]


def bilinear_interpolation_(imgs, x, y, xcyclic=False, all_dtype='float64'):
    '''
    imgs: met fields, shape=(1,lon,lat,vars)
    x: indices in lons, floats
    y: indices in lats, floats
    inter_img: interpolated values, shape = (npoints,vars)
    '''
    num_batches, width, height = imgs.shape[0], imgs.shape[1], imgs.shape[2]

    x0 = np.floor(x).astype('int32')
    x1 = x0 + 1
    y0 = np.floor(y).astype('int32')
    y1 = y0 + 1
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    a = get_pixel_value(imgs, x0, y1)
    b = get_pixel_value(imgs, x1, y1)
    c = get_pixel_value(imgs, x0, y0)
    d = get_pixel_value(imgs, x1, y0)

    slpx = x - x0
    slpy = y - y0
    wa = (1 - slpx) * (slpy)
    wb = (slpx) * (slpy)
    wc = (1 - slpx) * (1 - slpy)
    wd = (slpx) * (1 - slpy)

    inter_img = a * wa[:, None] + b * wb[:, None] + \
        c * wc[:, None] + d * wd[:, None]

    return inter_img


'''
or use tensorflow
'''


def bilinear_interpolation(imgs, x, y, xcyclic=False, all_dtype='float64'):
    '''
    imgs: met fields, shape=(1,lon,lat,vars)
    x: indices in lons, floats
    y: indices in lats, floats
    inter_img: interpolated values, shape = (npoints,vars)
    '''
    num_batches, width, height = imgs.shape[0], imgs.shape[1], imgs.shape[2]

    x0 = np.floor(x).astype('int32')
    x1 = x0 + 1
    y0 = np.floor(y).astype('int32')
    y1 = y0 + 1
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    a = get_pixel_value(imgs, x0, y1)
    b = get_pixel_value(imgs, x1, y1)
    c = get_pixel_value(imgs, x0, y0)
    d = get_pixel_value(imgs, x1, y0)

    wa = (x1 - x) * (y - y0)
    wb = (x - x0) * (y - y0)
    wc = (x1 - x) * (y1 - y)
    wd = (x - x0) * (y1 - y)

    inter_img = a * wa[:, None] + b * wb[:, None] + \
        c * wc[:, None] + d * wd[:, None]

    '''
    原方法中靠近左下的不插值
    但似乎不甚合理 如果不插值那么靠近四角的都不应该插值
    '''
    mask1 = np.ones(wa.shape) - 1
    mask2 = np.ones(wa.shape) - 1
    mask1[np.abs(x - x0) < delt] = 1
    mask2[np.abs(y - y0) < delt] = 1
    mask = mask1 * mask2
    indices = np.where(mask == 1)[0]
    inter_img[indices, :] = c[indices, :]
    return inter_img
# import tensorflow as tf
# def get_pixel_value(imgs, x, y):
#     '''
#     imgs: met fields, shape=(1,lon,lat,vars)
#     x: indices in lons
#     y: indices in lats
#     '''
#     imgs_ = tf.transpose(imgs,(1,2,0,3))
#     indices = tf.stack([x,y],axis=-1)
#     values = tf.gather_nd(imgs_,indices)
#     values = tf.transpose(values,(1,0,2))
#     # print(indices)

#     return values[0,:,:]


# def bilinear_interpolation(imgs, x, y,xcyclic=False):
#     '''
#     imgs: met fields, shape=(1,lon,lat,vars)
#     x: indices in lons, floats
#     y: indices in lats, floats
#     inter_img: interpolated values, shape = (npoints,vars)
#     '''
#     num_batches, width, height = imgs.shape[0], imgs.shape[1], imgs.shape[2]

#     imgs = tf.constant(imgs,dtype='float64')
#     x = tf.constant(x,dtype='float64')
#     y = tf.constant(y,dtype='float64')

#     x0 = tf.cast(tf.floor(x), 'int32')
#     x1 = x0 + 1
#     y0 = tf.cast(tf.floor(y), 'int32')
#     y1 = y0 + 1

#     if not xcyclic:
#         x0 = tf.clip_by_value(x0, 0, width - 1)
#         x1 = tf.clip_by_value(x1, 0, width - 1)
#         y0 = tf.clip_by_value(y0, 0, height - 1)
#         y1 = tf.clip_by_value(y1, 0, height - 1)
#     elif xcyclic:
#         imgs = tf.concat([imgs,imgs[0:1,0:1,:,:]],axis=1)
#         x0 = tf.clip_by_value(x0, 0, width)
#         x1 = tf.clip_by_value(x1, 0, width)
#         y0 = tf.clip_by_value(y0, 0, height)
#         y1 = tf.clip_by_value(y1, 0, height)
#     a = get_pixel_value(imgs, x0, y1)
#     b = get_pixel_value(imgs, x1, y1)
#     c = get_pixel_value(imgs, x0, y0)
#     # plt.imshow(c[0, :, :, :].astype('int32'))
#     # print(c[0, :, :, :])
#     d = get_pixel_value(imgs, x1, y0)

#     x0 = tf.cast(x0,'float64')
#     x1 = tf.cast(x1,'float64')
#     y0 = tf.cast(y0,'float64')
#     y1 = tf.cast(y1,'float64')

#     wa = (x1 - x) * (y - y0)
#     wb = (x - x0) * (y - y0)
#     wc = (x1 - x) * (y1 - y)
#     wd = (x - x0) * (y1 - y)

#     inter_img = a * wa[:,None] + b * wb[:,None] + c * wc[:,None] + d * wd[:,None]

#     return inter_img.numpy()

def linint2_point(nxi, xi, nyi, yi, fi, xcyclic, xo,
                  yo, fo_missing=np.nan, nopt=1):
    """
    双线性插值计算单点值 (对应 Fortran 的 linint2_point).
    参数:
      nxi, nyi     - xi, yi 数组长度 (需 >= 2)
      xi[0..nxi-1] - 自变量x坐标数组 (如经度), 要求单调递增
      yi[0..nyi-1] - 自变量y坐标数组 (如纬度), 要求单调递增
      fi[nxi, nyi] - 网格上的函数值数组
      xcyclic      - x方向是否周期 (True表示x是环状变量, 如经度)
      xo, yo       - 待插值的点坐标
      fo_missing   - 缺测值标记 (输出用)
      nopt         - 选项: =-1 时在缺测数据情况下采用距离加权平均，否则输出缺测
    返回:
      插值得到的函数值 (若无法插值则返回缺测标记 fo_missing)
    """
    # 输入有效性检查
    ier = 0
    if nxi < 2 or nyi < 2:
        ier = 1
    check_err(ier)
    ier = dmonoinc(xi)
    check_err(ier)
    ier = dmonoinc(yi)
    check_err(ier)
    if ier != 0:
        return fo_missing  # 数据不合法，返回缺测

    # 如果 x 方向周期性，则扩展xi和fi用于周期处理
    xi_arr = np.array(xi, dtype=float)
    fi_arr = np.array(fi, dtype=float)
    if xcyclic:
        # 规范化 xo 落入 [xi[0], xi[-1] + dx) 范围
        period = (xi_arr[-1] - xi_arr[0]) + (xi_arr[1] - xi_arr[0])
        xo = ((xo - xi_arr[0]) % period) + xi_arr[0]
        # 构造扩展坐标数组 (在两端各添加一个点)
        dx = xi_arr[1] - xi_arr[0]
        xiw = np.empty(nxi + 2, dtype=float)
        xiw[0] = xi_arr[0] - dx
        xiw[-1] = xi_arr[-1] + dx
        xiw[1:-1] = xi_arr
        # 构造扩展值数组 (两端复制原数组最后和最前的列)
        fixw = np.empty((nxi + 2, nyi), dtype=float)
        fixw[1:-1, :] = fi_arr
        fixw[0, :] = fi_arr[-1, :]   # 左端延拓：接续最后一个经度的数据
        fixw[-1, :] = fi_arr[0, :]   # 右端延拓：接续第一个经度的数据
        xi_use = xiw
        fi_use = fixw
        nxi_use = nxi + 2
    else:
        xi_use = xi_arr
        fi_use = fi_arr
        nxi_use = nxi

    # 在 xi_use 中查找 xo 所在的区间索引
    if xo < xi_use[0] or xo > xi_use[-1]:
        # xo 超出插值范围
        return fo_missing
    # 找到 xo 的相邻索引区间 [nx, nx+1)
    nx = int(np.searchsorted(xi_use, xo) - 1)
    if nx < 0:
        nx = 0
    if nx > nxi_use - 2:
        nx = nxi_use - 2

    # 在 yi 中查找 yo 所在的区间索引
    if yo < yi[0] or yo > yi[-1]:
        return fo_missing
    ny_index = int(np.searchsorted(yi, yo) - 1)
    if ny_index < 0:
        ny_index = 0
    if ny_index > nyi - 2:
        ny_index = nyi - 2

    # 取四个顶点值，检查是否缺测
    f11 = fi_use[nx, ny_index]
    f21 = fi_use[nx + 1, ny_index]
    f12 = fi_use[nx, ny_index + 1]
    f22 = fi_use[nx + 1, ny_index + 1]
    if (f11 == fo_missing or f21 == fo_missing or
            f12 == fo_missing or f22 == fo_missing):
        # 存在缺测值
        if nopt == -1:
            # 距离加权平均法近似
            # 这里简单采用周边非缺测值的平均作为替代
            vals = [f for f in [f11, f21, f12, f22] if f != fo_missing]
            return np.mean(vals) if vals else fo_missing
        else:
            return fo_missing

    # 进行双线性插值计算
    # 计算归一化位置 (相对于网格左下角的分数)
    t = (xo - xi_use[nx]) / (xi_use[nx + 1] - xi_use[nx])  # x方向比例
    u = (yo - yi[ny_index]) / (yi[ny_index + 1] - yi[ny_index])  # y方向比例

    # 按区域四点进行插值: 先对x插值，再对y插值
    f_low = f11 + t * (f21 - f11)    # 下边沿插值
    f_high = f12 + t * (f22 - f12)    # 上边沿插值
    fo = f_low + u * (f_high - f_low)  # 在y方向插值得到最终值

    return fo
