# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 22:48:01 2025

@author: 杨艺楠
"""

# wn.py - 波数计算模块
import numpy as np
from bs import BS, cal_ky
from scipy.ndimage import uniform_filter
import numba as nb

nb_para_dic = {
    'nopython': True,
    # 'fastmath':True,
    'cache': True,
}


class WN:
    """
    波数计算类 (对应 Fortran type(wn)):
    属性:
        bs       - BS 基本流场对象 (组合关系)
        nzwn     - 初始指定的 zonal 波数数量
        freq     - 波频率 (默认0表示定常波)
        zwn      - 初始 zonal 波数数组 (长度 nzwn)
        mwn      - 传播的 meridional 波数解数组 (维度 [nlon,nlat,nzwn,3])
        rootnum  - 每点实根个数 (维度 [nlon,nlat,nzwn])
        ug, vg   - 群速 (经、纬向) 数组 (维度 [nlon,nlat,nzwn,3])
    """

    def __init__(self, nx, ny, nzwn, freq=0.0):
        # 初始化 BS 对象并分配波参数数组
        self.bs = BS(nx, ny)
        self.nzwn = nzwn
        self.freq = freq if freq is not None else 0.0
        # 分配 zonal 波数数组并初始化为0
        self.zwn = np.zeros(nzwn, dtype=float)
        # 分配 meridional 波数和群速数组 (三根)
        # 维度: [nlon, nlat, nzwn, 3]
        self.mwn = np.zeros((nx, ny, nzwn, 3), dtype=float)
        self.rootnum = np.zeros((nx, ny, nzwn), dtype=int)
        self.ug = np.zeros((nx, ny, nzwn, 3), dtype=float)
        self.vg = np.zeros((nx, ny, nzwn, 3), dtype=float)

    def set_zwn(self, zwn_array):
        """设置初始的 zonal 波数数组 (长度必须为 nzwn)。"""
        if len(zwn_array) != self.nzwn:
            raise ValueError("Length of zwn_array must equal nzwn")
        self.zwn[:] = np.array(zwn_array, dtype=float)

    def set_freq(self, freq):
        """设置波的频率 (freq, 单位 rad/s)。"""
        self.freq = freq

    def cal_wave(self):
        """
        计算每个格点在各初始zonal波数下可传播的 meridional 波数 (mwn) 及对应群速度 (ug, vg)。
        对应 Fortran 的 cal_wave 子程序。
        """
        nx, ny = self.bs.nlon, self.bs.nlat
        # 遍历经纬网格
        for ix in range(nx):
            for iy in range(ny):
                # 取当前格点经纬坐标
                # 差值得到该点Mercator上的基本场和导数
                result = self.bs.cal_bs_mercator_point(
                    self.bs.lon[ix], self.bs.lat[iy])
                if result is None:
                    self._solve_at_grid_point(ix, iy, result)

    def _solve_at_grid_point(self, ix, iy, result):
        (fmu, fmv, _, _, _, _, fmqx, fmqy, *_rest) = result
        # 针对每个初始 zonal 波数计算 meridional 波数及群速
        for k, kz in enumerate(self.zwn):
            m_list, n_roots = cal_ky(fmu, fmv, fmqx, fmqy, self.freq, kz)
            self.rootnum[ix, iy, k] = n_roots
            for ir, m_val in enumerate(m_list):
                self.mwn[ix, iy, k, ir] = m_val
                # 计算对应每个根的群速度 ug, vg
                if not np.isnan(m_val):
                    self.ug[ix, iy, k, ir], self.vg[ix, iy, k, ir] = cal_ugvg(
                        fmu, fmv, fmqx, fmqy, kz, m_val)
                else:
                    self.ug[ix, iy, k, ir] = self.vg[ix, iy, k, ir] = 0.0


# 使用周围有效均值代替nan的波数、群速度：因为分母接近0时直接nan了================================


    def fast_fill_nan(arr, size=3):
        """
        用周围有效值均值替换 field 中的 NaN。
        对于每个 NaN，用其邻域中的非 NaN 值的平均填充。
        """
        mask = np.isnan(arr)
        arr_copy = np.where(mask, 0, arr)
        weight = (~mask).astype(float)

        arr_sum = uniform_filter(arr_copy, size=size, mode='wrap')
        weight_sum = uniform_filter(weight, size=size, mode='wrap')

        with np.errstate(invalid='ignore'):
            filled = arr.copy()
            filled[mask] = arr_sum[mask] / weight_sum[mask]
        return filled

    def postprocess(self): # 这个后处理目前没有进入主流程（保证与fortran一致）20250824
        self.ug = self.fast_fill_nan(self.ug)
        self.vg = self.fast_fill_nan(self.vg)
        self.mwn = self.fast_fill_nan(self.mwn)

# ============================================================================================
# 用3x3周围格点均值替换nan======================================================

    def fill_nan_by_local_mean(field, ix, iy, fallback=0.0):
        """用邻近格点的平均替换 field[ix, iy] 的 nan"""
        nx, ny = field.shape
        i_start, i_end = max(ix - 1, 0), min(ix + 2, nx)
        j_start, j_end = max(iy - 1, 0), min(iy + 2, ny)
        window = field[i_start:i_end, j_start:j_end]
        valid_vals = window[~np.isnan(window)]
        if valid_vals.size > 0:
            return np.mean(valid_vals)
        else:
            return fallback
# 在下面的diffun中进行==========================================================

    def clean(self):
        """释放 WN 对象的资源 (包括BS)。"""
        self.bs.clean()
        # 释放自身数组
        del self.mwn, self.rootnum, self.ug, self.vg

# 计算群速度的函数 (module-level), 对应 Fortran wn_mod 中的 cal_ugvg


def cal_ugvg_original(fu, fv, fqx, fqy, zwn, mwn):
    """
    计算群速 (ug, vg)。对应 Fortran cal_ugvg 子程序。
    输入:
      fu, fv   - Mercator 上风场 (u/cosφ, v/cosφ)
      fqx, fqy - 绝对涡度关于经度/纬度导数 (Mercator, 已缩放)
      zwn, mwn - 无量纲 zonal 波数和 meridional 波数 (k*R, m*R)
    输出:
      ug, vg   - 经度和纬度方向群速度 (单位: m/s)
    注：此函数负责单点计算，没有周围邻点的信息，不能进行空间平滑
    """
    if zwn == 0.0 or np.isnan(mwn) or np.isnan(
            fu) or np.isnan(fqx) or np.isnan(fqy):
        return 0, 0
    # 定义参数
    kap = mwn / zwn
    kap2 = kap * kap
    kap1 = 1.0 + kap2
    KK = zwn**2 * kap1  # k^2 * (1+kap^2)
    denom = KK * kap1
    # 计算群速 (源于频散关系对k和l的偏导)
    # 群速公式: ug = U + [ (1 - (m/k)^2) * (∂q/∂y) - 2*(m/k)*(∂q/∂x) ] / [k^2*(1+(m/k)^2)^2 ]
    # vg = V + [ 2*(m/k)*(∂q/∂y) + (1 - (m/k)^2)*(∂q/∂x) ] /
    # [k^2*(1+(m/k)^2)^2 ]

    # 不要除0（denom的作用），先返回nan给主程序处理，考虑使用周围均值填充
# =============================================================================
#     if abs(denom) < 1e-10: # 为确保与fortran一致暂时注释掉这一段，因为fortran没有这层保护处理 20250824
#         return np.nan, np.nan
# =============================================================================

    ug = fu + (((1 - kap2) * fqy) - (2 * kap * fqx)) / denom
    vg = fv + ((2 * kap * fqy) + ((1 - kap2) * fqx)) / denom
    # print('ug&vg',ug,vg)
    return ug, vg

'''
def cal_ugvg_numpy(fu, fv, fqx, fqy, zwn, mwn, min_val=1e-10):
    """
    fu,fv,fqx,fqy: (points,)
    mwn:           (3, points)
    返回 ug,vg:    (3, points)
    """
    # 与 Fortran 对齐：zwn==0 直接返回 0
    if zwn == 0:
        return np.zeros_like(mwn, dtype=float), np.zeros_like(mwn, dtype=float)

    # 公式（与 Fortran 等价，不做小分母阈值截断）
    kap1 = zwn * zwn - mwn * mwn          # k^2 - m^2   -> (3, points)
    kap2 = 2.0 * zwn * mwn                # 2 k m       -> (3, points)
    KK2  = zwn * zwn + mwn * mwn          # k^2 + m^2   -> (3, points)
    denom = KK2 * KK2                     # (k^2+m^2)^2 -> (3, points)

    # 纯广播求增量，再与 fu/fv 相加（fu[None,:] 会广播到 (3,points)）
    ug_raw = fu[None, :] + ((kap1 * fqy[None, :]) - (kap2 * fqx[None, :])) / denom
    vg_raw = fv[None, :] + ((kap1 * fqx[None, :]) + (kap2 * fqy[None, :])) / denom

    # 有效性掩膜：形状严格与 mwn 相同，避免布尔索引形状不匹配
    valid = (np.isfinite(mwn) &
             np.isfinite(fu)[None, :] & np.isfinite(fv)[None, :] &
             np.isfinite(fqx)[None, :] & np.isfinite(fqy)[None, :])

    # 与你“原函数”一致：无效处置 0（不返回 NaN、不做 3x3 回填）
    ug = np.where(valid, ug_raw, 0.0)
    vg = np.where(valid, vg_raw, 0.0)
    return ug, vg
'''


def cal_ugvg_numpy(fu, fv, fqx, fqy, zwn, mwn, min_val=1e-10): # 1e-10
    """
    add at 20250408 1:41
    计算群速 (ug, vg)。对应 Fortran cal_ugvg 子程序。
    输入:
      fu, fv   - Mercator 上风场 (u/cosφ, v/cosφ)
      fqx, fqy - 绝对涡度关于经度/纬度导数 (Mercator, 已缩放)
      zwn, mwn - 无量纲 zonal 波数和 meridional 波数 (k*R, m*R)
    输出:
      ug, vg   - 经度和纬度方向群速度 (单位: m/s)
    注：此函数负责单点计算，没有周围邻点的信息，不能进行空间平滑
    zwn: int
    fu,fv,fqx,fqy: (points,)
    mwn: (3,points)
    """
    ug = np.ones(mwn.shape) - 1
    vg = np.ones(mwn.shape) - 1
    if zwn == 0:
        pass
    else:
        nans = np.einsum('ij,j->ij', mwn * 0, fu * fqx * fqy * 0) + 1
        nans[np.isnan(nans)] = 0
        # 定义参数
        # kap = mwn / zwn
        # # kap[np.isnan(mwn)]=0 # 维持原始计算中mwn为nan时返回0的设定 kap=0时denom=zwn^2并不会是小值,但为0的设定与外层逻辑是矛盾的
        # #                      # 因为在外层又将mwn为nan的群速度设为了nan
        # kap2 = kap * kap
        # kap1 = 1.0 + kap2
        # KK = zwn**2 * kap1 # k^2 * (1+kap^2)
        # denom = KK * kap1
        # # 计算群速 (源于频散关系对k和l的偏导)
        # # 群速公式: ug = U + [ (1 - (m/k)^2) * (∂q/∂y) - 2*(m/k)*(∂q/∂x) ] / [k^2*(1+(m/k)^2)^2 ]
        # #           vg = V + [ 2*(m/k)*(∂q/∂y) + (1 - (m/k)^2)*(∂q/∂x) ] / [k^2*(1+(m/k)^2)^2 ]

        # # 不要除0（denom的作用），先返回nan给主程序处理，考虑使用周围均值填充
        # denom[np.abs(denom)<min_val] = np.nan

        # ug = fu + (((1 - kap2) * fqy) - (2 * kap * fqx)) / denom
        # vg = fv + ((2 * kap * fqy) + ((1 - kap2) * fqx)) / denom

        kap1 = zwn * zwn - mwn * mwn
        kap2 = 2 * zwn * mwn
        KK2 = zwn * zwn + mwn * mwn


        ug = fu + (kap1 * fqy - kap2 * fqx) / KK2**2
        vg = fv + (kap1 * fqx + kap2 * fqy) / KK2**2
        ug = ug * nans
        vg = vg * nans

    return ug, vg


sign1 = 'Tuple((f8[:,:,:],f8[:,:,:],f8[:,:,:]))(f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:])'
sign2 = 'Tuple((f4[:,:,:],f4[:,:,:],f4[:,:,:]))(f4[:,:,:],f4[:,:,:],f4[:,:,:],f4[:,:,:],f4[:,:,:],f4[:,:,:])'


@nb.jit([sign1], **nb_para_dic)  # ,sign2
def core_cal_ugvg_extent(fu, fv, fqx, fqy, zwn, mwn):
    # 定义参数
    kap = mwn / zwn
    # kap[np.isnan(mwn)]=0 # 维持原始计算中mwn为nan时返回0的设定 kap=0时denom=zwn^2并不会是小值,但为0的设定与外层逻辑是矛盾的
    #                      # 因为在外层又将mwn为nan的群速度设为了nan
    kap2 = kap * kap
    kap1 = 1.0 + kap2
    KK = zwn * zwn * kap1  # k^2 * (1+kap^2)
    denom = KK * kap1
    # 计算群速 (源于频散关系对k和l的偏导)
    # 群速公式: ug = U + [ (1 - (m/k)^2) * (∂q/∂y) - 2*(m/k)*(∂q/∂x) ] / [k^2*(1+(m/k)^2)^2 ]
    # vg = V + [ 2*(m/k)*(∂q/∂y) + (1 - (m/k)^2)*(∂q/∂x) ] /
    # [k^2*(1+(m/k)^2)^2 ]
    ug = fu + (((1. - kap2) * fqy) - (2. * kap * fqx)) / denom
    vg = fv + ((2. * kap * fqy) + ((1. - kap2) * fqx)) / denom
    ug = ug
    vg = vg
    # kap1 = zwn * zwn - mwn * mwn
    # kap2 = 2 * zwn * mwn
    # KK2  = zwn * zwn + mwn * mwn
    # denom = KK2

    # ug = fu + ( kap1 * fqy - kap2 * fqx ) / KK2**2
    # vg = fv + ( kap1 * fqx + kap2 * fqy ) / KK2**2
    # ug = ug
    # vg = vg

    return ug, vg, denom

'''
def cal_ugvg_extent(fu, fv, fqx, fqy, zwn_, mwn, min_val=1e-10):
    """
    fu,fv,fqx,fqy,zwn,mwn: (3, sources, waves)
    返回 ug,vg:            (3, sources, waves)
    """
    zwn = zwn_.copy()
    ug_raw, vg_raw, denom = core_cal_ugvg_extent(fu, fv, fqx, fqy, zwn, mwn)
    # 不做 1e-10 阈值截断（Fortran 没有）

    # 有效性掩膜：严格与 mwn 同形状
    valid = (np.isfinite(mwn) &
             np.isfinite(fu) & np.isfinite(fv) &
             np.isfinite(fqx) & np.isfinite(fqy))

    # 无效处置 0（与“原函数”/Fortran 无后处理的语义一致）
    ug = np.where(valid, ug_raw, 0.0)
    vg = np.where(valid, vg_raw, 0.0)
    return ug, vg
'''


def cal_ugvg_extent(fu, fv, fqx, fqy, zwn_, mwn, min_val=1e-10): # 1e-10
    """
    add at 20250408 1:41
    计算群速 (ug, vg)。对应 Fortran cal_ugvg 子程序。
    输入:
      fu, fv   - Mercator 上风场 (u/cosφ, v/cosφ)
      fqx, fqy - 绝对涡度关于经度/纬度导数 (Mercator, 已缩放)
      zwn, mwn - 无量纲 zonal 波数和 meridional 波数 (k*R, m*R)
    输出:
      ug, vg   - 经度和纬度方向群速度 (单位: m/s)
    注：此函数负责单点计算，没有周围邻点的信息，不能进行空间平滑
    fu,fv,fqx,fqy,zwn,mwn: (3,sources,waves)
    """
    ug = np.ones(mwn.shape) - 1
    vg = np.ones(mwn.shape) - 1
    zwn = zwn_.copy()
    # zwn[np.where(zwn_==0)]=np.nan

    # nans = mwn*fu*fqx*fqy*0+1
    # nans[np.isnan(nans)] = 0
    ug, vg, denom = core_cal_ugvg_extent(fu, fv, fqx, fqy, zwn, mwn)
    # 不要除0（denom的作用），先返回nan给主程序处理，考虑使用周围均值填充
    # denom[np.abs(denom)<min_val] = np.nan

    return ug, vg  # *nans#+0*denom


def cal_ugvg(fu, fv, fqx, fqy, zwn, mwn, mode='original'):
    if mode == 'original':
        return cal_ugvg_original(fu, fv, fqx, fqy, zwn, mwn)
    elif mode == 'numpy':
        return cal_ugvg_numpy(fu, fv, fqx, fqy, zwn, mwn)
    elif mode == 'extent':
        return cal_ugvg_extent(fu, fv, fqx, fqy, zwn, mwn)
