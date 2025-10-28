# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 22:48:27 2025

@author: 杨艺楠
"""

# wr.py - 波射线追踪模块

import numpy as np
from constants import day, hour, rad2deg, rearth, undef, deg2rad, pi
from bs import BS
from bs import cal_ky  # 也可直接使用 wn.cal_ky
from wn import cal_ugvg
from netCDF4 import Dataset
import numba as nb
import sys
from rkf45 import RK45


nb_para_dic = {
    'nopython': True,
    # 'fastmath': True,
    'cache': True,
}


def progress_bar(current, total, bar_length=50):
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write(
        f"\rprocess: [{arrow + spaces}] {int(round(percent * 100))}%")
    sys.stdout.flush()


signature1 = r'Tuple((' + r'f8[:,:,:],' * 7 + r'))' + \
    r'(f8[:],' + r'f8[:,:,:],' * 16 + r')'
signature2 = r'Tuple((' + r'f4[:,:,:],' * 7 + r'))' + \
    r'(f4[:],' + r'f4[:,:,:],' * 16 + r')'


@nb.jit([signature1], **nb_para_dic)
def core_diffun(freq, kx, ky, amp, fmu, fmv, fmux, fmuy, fmvx,
                fmvy, ug, vg, fmqxx, fmqxy, fmqyx, fmqyy, lat):
    '''
    core_diffun(freq, kx, ky, amp, fmu, fmv, fmux, fmuy, fmvx,
                fmvy, fmqx,fmqy, fmqxx, fmqxy, fmqyx, fmqyy, lat)
    '''
    ps = freq / kx
    up = fmu - ps
    kap = ky / kx
    kap2 = kap * kap
    kap1 = 1 + kap**2
    kk = kx**2 * kap1
    # ug = fmu + (((1.0 - kap2) * fmqy) - (2.0 * kap * fmqx)) / (kk * kap1)
    # vg = fmv + ((2.0 * kap * fmqy) + ((1.0 - kap2) * fmqx)) / (kk * kap1)
    # 计算波数演化 (Hamilton力学形式: dkx/dt, dky/dt)
    dzwn = - kx * ((fmux + kap * fmvx) + (kap * fmqxx - fmqyx) / kk)
    dmwn = - kx * ((fmuy + kap * fmvy) + (kap * fmqxy - fmqyy) / kk)
    # 计算振幅变化率 (damp = d(ln A)/dt, WKB能量守恒)
    damp1 = 2.0 * (fmux + fmvy + kap * (fmvx + fmuy)) / kap1
    damp2 = 2.0 * (kap * (fmqxx - fmqyy) + (kap2 - 1.0) * fmqxy) / (kk * kap1)
    damp3 = -2.0 * np.sin(lat) * fmv
    damp = damp1 + damp2 + damp3
    # 形成导数数组 (注意经度、纬度、波数等以地球半径归一化)
    dlon = ug
    dlat = vg * np.cos(lat)
    dkx = dzwn
    dky = dmwn
    damp_dt = damp * amp  # d(amp)/dt
    # 将空间分量按地球半径归一，得到每秒的变化率
    dlon /= rearth
    dlat /= rearth
    dkx /= rearth
    dky /= rearth
    damp_dt /= rearth
    # 额外将 ug, vg 作为状态变量导数存储 (按原Fortran实现，也除以Rearth)
    dug = ug  # / rearth
    dvg = vg  # / rearth
    return dlon, dlat, dkx, dky, damp_dt, dug, dvg


sign1 = 'f8[:,:,:,:](f8[:,:,:,:],f8[:,:,:,:],f8[:,:,:,:],f8[:,:,:,:],f8[:,:,:,:],f8[:])'
sign2 = 'f4[:,:,:,:](f4[:,:,:,:],f4[:,:,:,:],f4[:,:,:,:],f4[:,:,:,:],f4[:,:,:,:],f4[:])'


@nb.jit([sign1], **nb_para_dic)
def core_rk4_step(y, k1, k2, k3, k4, dt):
    temp = y
    ks = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    temp[0:5] = y[0:5] + ks[0:5]
    temp[5::] = ks[5::] / dt
    return temp

@nb.jit('f8[:,:,:](f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:])', **nb_para_dic)
def cal_dis(lon_curr,lat_curr,lon_prev,lat_prev):

    # 计算经纬度差值 (弧度)
    dlon = lon_curr - lon_prev # 对应 Haversine 公式中的 (lambda2 - lambda1)
    dlat = lat_curr - lat_prev # 对应 Haversine 公式中的 (phi2 - phi1)

    # Haversine 公式计算步骤
    # 计算 a
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_prev) * np.cos(lat_curr) * np.sin(dlon / 2.0)**2

    # 计算 c (角距离)
    # 使用 np.arctan2 比 np.arccos 在数值上更稳定，尤其当两点非常接近时
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return np.abs(c)

class WR:
    """
    波射线追踪类 (对应 Fortran type(wr)):
    属性:
      bs         - 基本流场 BS 对象 (组合)
      nzwn       - 初始 zonal 波数数量
      zwn        - zonal 波数数组 (长度 nzwn)
      freq       - 波频率
      nsource    - 波源点总数
      source_lon, source_lat - 波源位置数组 (单位: 弧度, 长度 nsource)
      tstep      - 时间步长 (秒)
      ttotal     - 总积分时间 (秒)
      nt         - 积分步数 (int(ttotal/tstep) + 1)
      rlon, rlat - 射线位置 (经度、纬度) 随时间演变 (维度 [nt, 3, nsource, nzwn]与fortran一致)
      rzwn, rmwn - 射线局地波数 (zonal, meridional) 随时间演变 (同上维度)
      ramp       - 射线振幅随时间演变 (同上维度)
      rug, rvg   - 射线群速度随时间演变 (经、纬方向, 同上维度)
    """

    def __init__(self, nzwn, nsource, tstep=1. * hour, ttotal=20. *
                 day, freq=0, cal_dtype='float64', read_dtype='float32',
                 rtol=1e-6, atol=1e-6, cut_off=0.1, nx=None, ny=None, ncfile=None,
                 MinStepFactor=1e-3):
        # 初始化基本流场
        self.all_dtype = cal_dtype
        if (nx is None) or (ny is None):
            if ncfile is None:
                raise ValueError('ncfile is need')
            else:
                ny, nx = self.get_defualt_nlon_and_nlat(ncfile)
        self.bs = BS(nx, ny, read_dtype=read_dtype, cal_dtype=cal_dtype)
        # day, hour, rad2deg, rearth, undef, deg2rad, pi = np.array([day, hour, rad2deg, rearth, undef, deg2rad, pi], dtype = cal_dtype)
        # 设置可选参数默认值: tstep默认1小时, ttotal默认20天, freq默认0
        self.tstep = np.array([tstep], dtype=self.all_dtype)
        self.ttotal = ttotal
        self.freq = np.array([freq], dtype=self.all_dtype)
        self.nzwn = nzwn
        # 分配数组
        self.zwn = np.zeros(nzwn, dtype=self.all_dtype)
        self.nsource = nsource
        self.source_lon = np.zeros(nsource, dtype=self.all_dtype)
        self.source_lat = np.zeros(nsource, dtype=self.all_dtype)
        # 计算步数 nt
        self.nt = int(self.ttotal / self.tstep[0]) + 1
        # 分配射线跟踪数组: shape = (nzwn, nsource, 3, nt)
        # 第三维大小为3，对应每初始波数可能存在的3条射线 (3个 m 根)
        shape = (self.nt, 3, nsource, nzwn)
        self.rlon = np.full(shape, undef, dtype=self.all_dtype)
        self.rlat = np.full(shape, undef, dtype=self.all_dtype)
        self.rzwn = np.full(shape, undef, dtype=self.all_dtype)
        self.rmwn = np.full(shape, undef, dtype=self.all_dtype)
        self.ramp = np.full(shape, undef, dtype=self.all_dtype)
        self.rug = np.full(shape, undef, dtype=self.all_dtype)
        self.rvg = np.full(shape, undef, dtype=self.all_dtype)
        self.rtol = rtol
        self.atol = atol
        self.cut_off = cut_off  * self.tstep / 3600.
        self.MinStepFactor = MinStepFactor

    def get_defualt_nlon_and_nlat(self, ncfile):
        '''
        如果没有指定mm和nn
        通过读取nc文件确定它们
        如果文件中存在lon和lat，使用它们的维度
        如果不存在lon或lat，使用风场的维度，默认-1维为lon，-2维为lat，并给出warning
        '''
        import netCDF4 as nc
        ds = nc.Dataset(ncfile)

        temp_u = np.array(ds.variables['u'][:], dtype=self.all_dtype)

        # 自动识别纬度、经度变量
        lat_candidates = ['lat', 'latitude', 'Lat', 'Latitude']
        lon_candidates = ['lon', 'longitude', 'Lon', 'Longitude']

        lat_data = None
        lon_data = None

        for name in lat_candidates:
            if name in ds.variables:
                lat_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                break
        for name in lon_candidates:
            if name in ds.variables:
                lon_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                break
        message = ''
        if (lat_data is None) or (lon_data is None):
            nlat, nlon = temp_u.shape[-2], temp_u.shape[-1]
            message = '!!!WARNING: Using u.shape[-2] and u.shape[-1] as nlat and nlon!!!'
            print(message)
        else:
            nlat, nlon = lat_data.shape[0], lon_data.shape[0]
        ds.close()
        return nlat, nlon

    def set_zwn(self, zwn_array):
        """设置初始 zonal 波数数组。"""
        if len(zwn_array) != self.nzwn:
            raise ValueError("Length of zwn_array must equal nzwn")
        self.zwn[:] = np.array(zwn_array, dtype=self.all_dtype)

    def set_freq(self, freq):
        """设置波频率。"""
        self.freq = np.array([freq], dtype=self.all_dtype)

    def set_source_array(self, lon_list, lat_list):
        """
        直接设置源点数组 (lon_list, lat_list)。
        lon/lat 单位为度，将转换为弧度后存储。
        """
        if len(lon_list) != self.nsource or len(lat_list) != self.nsource:
            raise ValueError("Source list length mismatch nsource")
        self.source_lon = np.array(lon_list,
                                   dtype=self.all_dtype) * (1.0 * deg2rad)
        self.source_lat = np.array(lat_list,
                                   dtype=self.all_dtype) * (1.0 * deg2rad)

    def set_source_matrix(self, SW_lon, SW_lat, dlon, dlat, nnx, nny):
        """
        设置源点为规则网格: 以 (SW_lon, SW_lat) 为起点，间隔 dlon, dlat (单位度)，
        网格大小 nnx x nny，总数应等于 nsource。
        """
        # 确认源点数匹配
        if nnx * nny != self.nsource:
            raise ValueError("nsource != nnx * nny, matrix size mismatch!")
        # 检查纬度范围
        if SW_lat + (nny - 1) * dlat > 89.0:
            raise ValueError("source latitude out of -90~90 range!")
        # 规范化起始经度到 [0, 360)
        SW_lon = SW_lon % 360.0
        idx = 0
        for iy in range(nny):
            for ix in range(nnx):
                lon_deg = SW_lon + ix * dlon
                lat_deg = SW_lat + iy * dlat
                # 经度规范化到 [0, 360)
                lon_deg = lon_deg % 360.0
                self.source_lon[idx] = lon_deg * deg2rad
                self.source_lat[idx] = lat_deg * deg2rad
                idx += 1

    def ray_info(self):
        """输出波射线追踪的初始信息"""
        print(
            "==============================================================================")
        print(" WNWR Package: Barotropic Horizontal Rossby Wave Ray Tracing Information ")
        print(
            f" Shape of the Basic Flow (nlon x nlat): {self.bs.nlon} x {self.bs.nlat}")
        print(f" Initial Zonal Wave Numbers (nzwn): {self.nzwn}")
        print(" " * 15 + " ".join(f"{z:.1f}" for z in self.zwn))
        print(f" Source Locations (total {self.nsource} points):")
        # 打印每个源点经纬度（度）
        for i in range(self.nsource):
            lon_deg = self.source_lon[i] * rad2deg
            lat_deg = self.source_lat[i] * rad2deg
            print(" " * 15 + f"{lon_deg:7.2f}, {lat_deg:7.2f}")
        print(f" Time Step (s): {self.tstep[0]:.1f}")
        print(f" Total Integration Time (day): {self.ttotal/day:.1f}")
        print(f" Total Steps (nt): {self.nt}")
        print(
            "==============================================================================")

    def ray_initial_original(self):
        """
        初始化射线的起始状态:
         - 将所有射线的初始位置设置为源点位置
         - 计算每个源点每个初始k对应的 meridional 波数根 (最多3个)，赋初始 k, l, 振幅等
         - 计算初始群速度
        """
        # 为方便，将 initial 状态索引用 idx0 表示 (Python 0号索引对应 Fortran 初始状态第1步)
        idx0 = 0
        # 对每个源点:
        for isrc in range(self.nsource):
            # 将所有波数和根的经纬初始位置设为源点
            for iz in range(self.nzwn):
                for ir in range(3):
                    self.rlon[idx0, ir, isrc, iz] = self.source_lon[isrc]
                    self.rlat[idx0, ir, isrc, iz] = self.source_lat[isrc]
            # 差值基本场到源点位置
            lon0 = self.source_lon[isrc]
            lat0 = self.source_lat[isrc]
            res = self.bs.cal_bs_mercator_point(lon0, lat0)
            if res is None:
                continue
            (fmu, fmv, fmux, fmuy, fmvx, fmvy,
             fmqx, fmqy, fmqxx, fmqxy, fmqyx, fmqyy,
             fmqxxx, fmxxy, fmqxyy, fmqyyy, fmqyxx, fmqyyx) = res
            # 对每个初始 zonal 波数求 m 并初始化状态
            for iz in range(self.nzwn):
                kz = self.zwn[iz]
                # 设置局地 k (zwn) 初值
                for ir in range(3):
                    self.rzwn[idx0, ir, isrc, iz] = kz
                # 计算 meridional 波数根
                m_list, rootnum = cal_ky(fmu, fmv, fmqx, fmqy, self.freq, kz)
                for ir in range(3):
                    m_val = m_list[ir]
                    self.rmwn[idx0, ir, isrc, iz] = m_val
                    # 初始振幅，可设为1 (若需要更复杂的初始振幅和相位，可在此调整)
                    self.ramp[idx0, ir, isrc, iz] = 1.0 if (
                        not np.isnan(m_val)) else undef
                    # 计算初始群速度并存储
                    if not np.isnan(m_val):
                        ug0, vg0 = cal_ugvg(fmu, fmv, fmqx, fmqy, kz, m_val)
                        if np.isnan(ug0) or abs(
                                self.rlat[idx0, ir, isrc, iz]) >= pi / 2:
                            self.bs.terminate_ray(
                                self.rlon,
                                self.rlat,
                                self.ramp,
                                self.rug,
                                self.rvg,
                                self.rzwn,
                                self.rmwn,
                                iz,
                                isrc,
                                ir,
                                idx0)
                            continue
                        self.rug[idx0, ir, isrc, iz] = ug0
                        self.rvg[idx0, ir, isrc, iz] = vg0
                    else:
                        self.rug[idx0, ir, isrc, iz] = undef
                        self.rvg[idx0, ir, isrc, iz] = undef

    def ray_initial_numpy(self, root_method='Fortran'):
        """
        NumPy 向量化的初始化：
         - 把 (lon, lat) 初始位置设置为源点
         - 计算三根 m，并初始化 k、振幅、群速
         - 新增：对不可传播根（rmwn 为 NaN），把 ramp 同步设为 NaN（与 Fortran 语义一致）
        """
        idx0 = 0
    
        # 1) 初始位置：所有根/所有源点/所有 k 的经纬度 = 源点位置
        self.rlon[idx0, :, :, :] = 1.0
        self.rlat[idx0, :, :, :] = 1.0
        self.rlon[idx0, :, :, :] *= self.source_lon[None, :, None]
        self.rlat[idx0, :, :, :] *= self.source_lat[None, :, None]
    
        # 2) 源点上背景场（一次性矢量化插值）
        lon0 = self.source_lon[:]  # (nsource,)
        lat0 = self.source_lat[:]
        res = self.bs.cal_bs_mercator_point(lon0, lat0, mode='numpy')
        (fmu, fmv, fmux, fmuy, fmvx, fmvy,
         fmqx, fmqy, fmqxx, fmqxy, fmqyx, fmqyy,
         fmqxxx, fmxxy, fmqxyy, fmqyyy, fmqyxx, fmqyyx) = res
    
        # 3) 初始 k：对每个 iz，三根位置都赋同一个初值 kz
        self.rzwn[idx0, :, :, :] = 1.0
        self.rzwn[idx0, :, :, :] *= self.zwn[None, None, :]
    
        # 4) 对每个初始 k 计算 m 的三根，并初始化 ramp、ug、vg
        for iz in range(self.nzwn):
            kz = self.zwn[iz]
    
            # 计算该 kz 下、所有源点的 3 个 m 根（shape 期望为 (3, nsource)）
            # 你的 cal_ky 会根据 root_method='Fortran' 或 'numpy' 返回 m_list 与 rootnum
            m_list, rootnum = cal_ky(
                fmu, fmv, fmqx, fmqy, self.freq, kz, iz=iz, mode='numpy', root_method=root_method
            )
            # 统一转成 (3, nsource) -> 我们在 rmwn 中的目标形状是 (3, nsource, nzwn)
            # 你现有实现返回 m_list 形状常见为 (nsource, 3)，因此转置：
            m_val = np.transpose(m_list, (1, 0))  # (3, nsource)
    
            # 写入 rmwn 初始值
            self.rmwn[idx0, :, :, iz] = m_val
    
            # 初始振幅：默认赋 1.0；随后对不可传播根（NaN）置 NaN（补丁）
            self.ramp[idx0, :, :, iz] = 1.0
            mask_nan = np.isnan(m_val)
            self.ramp[idx0, :, :, iz][mask_nan] = np.nan
    
            # 初始群速 ug, vg（与 Fortran 同公式）
            ug0, vg0 = cal_ugvg(fmu, fmv, fmqx, fmqy, kz, m_val, mode='numpy')
            self.rug[idx0, :, :, iz] = ug0
            self.rvg[idx0, :, :, iz] = vg0


    def load_init_from_precal_nc(self, ncfile):
        '''
        使用fortran计算的初始信息，用于debug
        '''
        import netCDF4 as nc
        ds = nc.Dataset(ncfile)
        names = ['rlon', 'rlat', 'rzwn', 'rmwn', 'ramp', 'rug', 'rvg']
        for name in names:
            if name not in ['rlon', 'rlat']:
                temp = np.array(ds.variables[name][:], dtype=self.all_dtype)
            elif name in ['rlon', 'rlat']:
                temp = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype) / 180 * pi
            temp[1::, :, :, :] = np.nan
            temp[temp == 999.] = np.nan
            setattr(self, name, temp)
        ds.close()

    def ray_initial(self, mode='original', root_method='Fortran'):
        if mode == 'original':
            self.ray_initial_original()
        elif mode == 'numpy':
            self.ray_initial_numpy(root_method=root_method)

    def diffun_original(self, y):
        """
        计算状态变量 y 的导数 dy/dt，用于 Runge-Kutta 积分。
        y 为长度7的数组: [lon, lat, kx, ky, amp, ug, vg]
        返回相同长度的导数组 dk (或在不继续积分时返回 None)。
        """
        lon, lat, kx, ky, amp = y[0], y[1], y[2], y[3], y[4]
        # 检查纬度范围限制
        if abs(lat) >= 0.5 * pi:
            # 超过可追踪范围，停止
            print("the latitude out of limit")
            return None
        # 计算当前位置背景场变量 (Mercator)
        res = self.bs.cal_bs_mercator_point(lon, lat)
        if res is None:
            return None
        (fmu, fmv, fmux, fmuy, fmvx, fmvy,
         fmqx, fmqy, fmqxx, fmqxy, fmqyx, fmqyy,
         fmqxxx, fmxxy, fmqxyy, fmqyyy, fmqyxx, fmqyyx) = res
        # 计算当前群速度和波数变化率
        # 注意: 这里使用上一步算出的 ky, kx 来计算
        # 如果 ky 过大，停止积分
        if abs(ky) >= 100.0:
            return None
        # 调用 cal_vars 计算群速 (ug, vg) 及 kx, ky, amp 的变化率
        zwn = kx  # * rearth
        mwn = ky  # * rearth
        ug, vg = cal_ugvg(fmu, fmv, fmqx, fmqy, zwn, mwn)
        # print(ug,vg)
        # if np.isnan(ug) or np.isnan(vg):
        # # 使用邻近均值替代群速度，避免波射线中断====================================
        #     ix = np.searchsorted(self.bs.lon, lon) - 1
        #     iy = np.searchsorted(self.bs.lat, lat) - 1
        #     ug = self.wn.fill_nan_by_local_mean(self.wn.ug[:, :, self.kid, self.rid], ix, iy)
        #     vg = self.wn.fill_nan_by_local_mean(self.wn.vg[:, :, self.kid, self.rid], ix, iy)
        kap = ky / kx
        kap2 = kap * kap
        kap1 = 1 + kap**2
        kk = kx**2 * kap1

        # ug = fmu + (((1.0 - kap2) * fmqy) - (2.0 * kap * fmqx)) / (kk * kap1)
        # vg = fmv + ((2.0 * kap * fmqy) + ((1.0 - kap2) * fmqx)) / (kk * kap1)

        # 计算波数演化 (Hamilton力学形式: dkx/dt, dky/dt)
        dzwn = - kx * ((fmux + kap * fmvx) + (kap * fmqxx - fmqyx) / kk)
        dmwn = - kx * ((fmuy + kap * fmvy) + (kap * fmqxy - fmqyy) / kk)
        # 计算振幅变化率 (damp = d(ln A)/dt, WKB能量守恒)
        damp1 = 2.0 * (fmux + fmvy + kap * (fmvx + fmuy)) / kap1
        damp2 = 2.0 * (kap * (fmqxx - fmqyy) +
                       (kap2 - 1.0) * fmqxy) / (kk * kap1)
        damp3 = -2.0 * np.sin(lat) * fmv
        damp = damp1 + damp2 + damp3
        # 形成导数数组 (注意经度、纬度、波数等以地球半径归一化)
        dlon = ug
        dlat = vg * np.cos(lat)
        dkx = dzwn
        dky = dmwn
        damp_dt = damp * amp  # d(amp)/dt
        # 将空间分量按地球半径归一，得到每秒的变化率
        dlon /= rearth
        dlat /= rearth
        dkx /= rearth
        dky /= rearth
        damp_dt /= rearth
        # 额外将 ug, vg 作为状态变量导数存储 (按原Fortran实现，也除以Rearth)
        dug = ug / rearth
        dvg = vg / rearth
        return np.array([dlon, dlat, dkx, dky, damp_dt, dug, dvg], dtype=float)

    def diffun_numpy(self, y):
        """
        计算状态变量 y 的导数 dy/dt（NumPy 向量化版本，逐射线判定失效）
        y 形状: (7, 3, nsource, nzwn) 依次是 [lon, lat, kx, ky, amp, ug, vg]
        返回:
          derivs: 与 y 同形状的导数数组
          err_mask: 形状与 y[0] 相同的布尔掩膜(True 表示该条射线应终止/冻结)
        """
        # 拆分状态量
        lon = y[0]  # (3, nsource, nzwn)
        lat = y[1]
        kx  = y[2]
        ky  = y[3]
        amp = y[4]
    
        # -------- 逐射线失效判据（而非 .all()） --------
        lat_fail = (np.abs(lat) >= 0.5 * pi)   # 越界（±90°）
        ky_fail  = (np.abs(ky)  >= 100.0)      # |m|过大视作不可传播
        err_mask = (lat_fail | ky_fail)        # True 表示这条射线此刻应停
    
        # 为避免失效条目参与后续计算：把 ky 设成 NaN（位置/背景插值仍允许）
        ky_safe = ky.copy()
        ky_safe[err_mask] = np.nan
    
        # -------- 背景场插值（向量化）--------
        # 将 (3, nsource, nzwn) 摊平为 (-1,) 做一次性插值，再 reshape 回去
        flat_shape = lon.shape
        lon_flat = lon.reshape(-1)
        lat_flat = lat.reshape(-1)
        res = self.bs.cal_bs_mercator_point(lon_flat, lat_flat, mode='numpy')
        (fmu, fmv, fmux, fmuy, fmvx, fmvy,
         fmqx, fmqy, fmqxx, fmqxy, fmqyx, fmqyy,
         fmqxxx, fmxxy, fmqxyy, fmqyyy, fmqyxx, fmqyyx) = res
    
        # reshape 回 (3, nsource, nzwn)
        fmu   = fmu.reshape(flat_shape)
        fmv   = fmv.reshape(flat_shape)
        fmux  = fmux.reshape(flat_shape)
        fmuy  = fmuy.reshape(flat_shape)
        fmvx  = fmvx.reshape(flat_shape)
        fmvy  = fmvy.reshape(flat_shape)
        fmqx  = fmqx.reshape(flat_shape)
        fmqy  = fmqy.reshape(flat_shape)
        fmqxx = fmqxx.reshape(flat_shape)
        fmqxy = fmqxy.reshape(flat_shape)
        fmqyx = fmqyx.reshape(flat_shape)
        fmqyy = fmqyy.reshape(flat_shape)
        # 其余三阶导数此处不需要，略
    
        # -------- 群速度（与 Fortran 同公式；mode='extent' 为你的批量版实现）--------
        ug, vg = cal_ugvg(fmu, fmv, fmqx, fmqy, kx, ky_safe, mode='extent')
    
        # -------- 右端项（不改变你已有的 core_diffun 逻辑）--------
        dlon, dlat, dkx, dky, damp_dt, dug, dvg = core_diffun(
            self.freq, kx, ky_safe, amp,
            fmu, fmv, fmux, fmuy, fmvx, fmvy,
            ug, vg, fmqxx, fmqxy, fmqyx, fmqyy, lat
        )
    
        # 对失效条目：显式置 NaN，表示“这一条已停止/冻结”
        for arr in (dlon, dlat, dkx, dky, damp_dt, dug, dvg):
            arr[err_mask] = np.nan
    
        derivs = np.array([dlon, dlat, dkx, dky, damp_dt, dug, dvg], dtype=self.all_dtype)
        return derivs, err_mask


    def diffun(self, y, mode='original'):
        if mode == 'original':
            return self.diffun_original(y)
        elif mode == 'numpy':
            return self.diffun_numpy(y)

    def rk4_step_original(self, y):
        """对状态变量 y 执行单步四阶Runge-Kutta积分，返回积分后的新状态。"""
        dt = self.tstep
        k1 = self.diffun(y)
        if k1 is None:
            return None
        k2 = self.diffun(y + 0.5 * dt * k1)
        if k2 is None:
            return None
        k3 = self.diffun(y + 0.5 * dt * k2)
        if k3 is None:
            return None
        k4 = self.diffun(y + dt * k3)
        if k4 is None:
            return None
        # RK4 更新公式: y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4_step_numpy(self, y):
        """
        固定步长 RK4（NumPy 向量化），逐射线推进：
        - 对仍有效的条目执行完整 RK4 更新
        - 对失效条目保持冻结（不再更新，维持 NaN/原状）
        返回:
          y_next: 与 y 同形状的新状态
          err_mask: 这一时间层评估得到的“应终止/冻结”掩膜
        """
        dt = self.tstep  # 标量（以秒计），dtype 与 self.all_dtype 一致
        k1, mask1 = self.diffun(y, mode='numpy')
    
        # 预先复制一份，默认保持“原状”（失效条目会继续保持 NaN/原值）
        y_next = y.copy()
    
        # 只有“仍有效”的条目才推进
        valid1 = ~mask1
        if np.any(valid1):
            # k2
            k2, mask2 = self.diffun(y + 0.5 * dt * k1, mode='numpy')
            # k3
            k3, mask3 = self.diffun(y + 0.5 * dt * k2, mode='numpy')
            # k4
            k4, mask4 = self.diffun(y + dt * k3, mode='numpy')
    
            # “四阶段都有效”的条目，才允许本步写回
            valid_all = valid1 & (~mask2) & (~mask3) & (~mask4)
    
            # 计算完整 RK4 提案解（与你原有 core_rk4_step 完全一致）
            y_prop = core_rk4_step(y, k1, k2, k3, k4, dt)
    
            # 只把 valid_all 的位置写回；其余位置保持原值（冻结）
            # 注意：valid_all 形状是 (3, nsource, nzwn)
            idx = np.where(valid_all)
            # y 的第 0 轴是变量维（7 个变量），其余轴与 valid_all 对齐
            y_next[(slice(None),) + idx] = y_prop[(slice(None),) + idx]
    
        # err_mask 返回这一时刻的“需要冻结/终止”的条目（用于上层统计/可视化）
        err_mask = mask1
        return y_next, err_mask


    def rk4_step(self, y, mode='original'):
        if mode == 'original':
            return self.rk4_step_original(y)
        elif mode == 'numpy':
            return self.rk4_step_numpy(y)

    def core_ray_run_original(self):
        for isrc in range(self.nsource):
            lon0_deg = self.rlon[0, 0, isrc, 0] * rad2deg
            lat0_deg = self.rlat[0, 0, isrc, 0] * rad2deg
            print(f"Source {isrc+1}: Lon={lon0_deg:.3f}°, Lat={lat0_deg:.3f}°")

            for iz in range(self.nzwn):
                for ir in range(3):
                    if np.isnan(self.rmwn[0, ir, isrc, iz]):
                        continue
                    y = np.array([
                        self.rlon[0, ir, isrc, iz],
                        self.rlat[0, ir, isrc, iz],
                        self.rzwn[0, ir, isrc, iz],
                        self.rmwn[0, ir, isrc, iz],
                        self.ramp[0, ir, isrc, iz],
                        self.rug[0, ir, isrc, iz],
                        self.rvg[0, ir, isrc, iz]
                    ], dtype=float)
                    for it in range(self.nt - 1):
                        print(it)
                        result = self.rk4_step(y)
                        if result is None or np.isnan(
                                result).any() or abs(result[1]) >= pi / 2:
                            self.bs.terminate_ray(
                                self.rlon,
                                self.rlat,
                                self.ramp,
                                self.rug,
                                self.rvg,
                                self.rzwn,
                                self.rmwn,
                                iz,
                                isrc,
                                ir,
                                it)
                            break
                        lon_new, lat_new = result[0], result[1]
                        kx_new, ky_new = result[2], result[3]
                        res = self.bs.cal_bs_mercator_point(lon_new, lat_new)
                        if res is None:
                            self.bs.terminate_ray(
                                self.rlon,
                                self.rlat,
                                self.ramp,
                                self.rug,
                                self.rvg,
                                self.rzwn,
                                self.rmwn,
                                iz,
                                isrc,
                                ir,
                                it)
                            break
                        (fmu, fmv, fmux, fmuy, fmvx, fmvy,
                         fmqx, fmqy, *_) = res
                        ug_new, vg_new = cal_ugvg(
                            fmu, fmv, fmqx, fmqy, kx_new, ky_new)
                        result[5] = ug_new
                        result[6] = vg_new
                        y = result
                        self.rlon[it + 1, ir, isrc, iz] = y[0]
                        # print(y[0]/pi*180)
                        self.rlat[it + 1, ir, isrc, iz] = y[1]
                        # print(y[1]/pi*180)
                        self.rzwn[it + 1, ir, isrc, iz] = y[2]
                        self.rmwn[it + 1, ir, isrc, iz] = y[3]
                        self.ramp[it + 1, ir, isrc, iz] = y[4]
                        self.rug[it + 1, ir, isrc, iz] = y[5]
                        self.rvg[it + 1, ir, isrc, iz] = y[6]

    def core_ray_run_numpy(self):
        y = np.array([
            self.rlon[0, :, :, :],
            self.rlat[0, :, :, :],
            self.rzwn[0, :, :, :],
            self.rmwn[0, :, :, :],
            self.ramp[0, :, :, :],
            self.rug[0, :, :, :],
            self.rvg[0, :, :, :]
        ], dtype=self.all_dtype)

        for it in range(self.nt - 1):
            progress_bar(it, self.nt, bar_length=50)
            result, err = self.rk4_step(y, mode='numpy')
            # if err == 1:
            #     break
            lon_new, lat_new = result[0], result[1]
            kx_new, ky_new = result[2], result[3]
            amp_new = result[4]
            nan_indices = np.where(np.abs(lat_new) >= 0.5 * pi)
            lon_new[nan_indices] = np.nan
            lat_new[nan_indices] = np.nan
            kx_new[nan_indices] = np.nan
            ky_new[nan_indices] = np.nan
            amp_new[nan_indices] = np.nan
            # print((np.isnan(lon_new)).all())
            ddis = cal_dis(lon_new,lat_new,self.rlon[it],self.rlat[it])
            nan_indices = np.where(np.abs(ddis) >= self.cut_off)
            lon_new[nan_indices] = np.nan
            lat_new[nan_indices] = np.nan
            kx_new[nan_indices] = np.nan
            ky_new[nan_indices] = np.nan
            amp_new[nan_indices] = np.nan
            if (np.isnan(lon_new)).all() or (np.abs(lat_new) > 0.5 * pi).all():
                break
            res = self.bs.cal_bs_mercator_point(
                lon_new.reshape(-1), lat_new.reshape(-1), mode='numpy')

            (fmu, fmv, fmux, fmuy, fmvx, fmvy,
                fmqx, fmqy, *_) = res
            ug_new, vg_new = cal_ugvg(fmu.reshape(result[0].shape),
                                      fmv.reshape(result[0].shape),
                                      fmqx.reshape(result[0].shape),
                                      fmqy.reshape(result[0].shape),
                                      kx_new, ky_new, mode='extent')
            # print(ug_new,vg_new)
            self.rlon[it + 1, :, :, :] = lon_new
            # print(y[0][0,0,0]/pi*180)
            self.rlat[it + 1, :, :, :] = lat_new
            # print(y[1][0,0,0]/pi*180)
            self.rzwn[it + 1, :, :, :] = kx_new
            self.rmwn[it + 1, :, :, :] = ky_new
            self.ramp[it + 1, :, :, :] = amp_new
            self.rug[it + 1, :, :, :] = ug_new
            self.rvg[it + 1, :, :, :] = vg_new
            y = np.array([
                self.rlon[it + 1, :, :, :],
                self.rlat[it + 1, :, :, :],
                self.rzwn[it + 1, :, :, :],
                self.rmwn[it + 1, :, :, :],
                self.ramp[it + 1, :, :, :],
                self.rug[it + 1, :, :, :],
                self.rvg[it + 1, :, :, :]
            ], dtype=self.all_dtype)

    def core_ray_run_rk45(self):
        y_ = np.array([
            self.rlon[0, :, :, :],
            self.rlat[0, :, :, :],
            self.rzwn[0, :, :, :],
            self.rmwn[0, :, :, :],
            self.ramp[0, :, :, :],
            # self.rug[0, :, :, :],
            # self.rvg[0, :, :, :]
        ], dtype=self.all_dtype)
        # dt = self.tstep
        t0 = 0
        t_end = self.ttotal
        n_vars = y_.shape[0]
        y = np.asarray(y_, dtype=self.all_dtype).reshape((n_vars, -1))
        # print(y.shape)

        def fun(t, y_):
            # print(y_.shape)
            y = y_.reshape((n_vars, -1, 1, 1))
            dx, _ = self.diffun_numpy(y)
            dx = dx[0:5]
            return dx.reshape((n_vars, -1))
        # --- Initialization ---
        # t = t0
        solver = RK45(fun, t0, y, self.tstep,#t_end,
                      rtol=self.rtol, atol=self.atol,
                      all_dtype=self.all_dtype, Global_Minstep=self.MinStepFactor*self.tstep)
        # --- Main Integration Loop ---

        # i = 0
        t_eval = np.arange(self.nt) * self.tstep
        if t_eval[-1] > self.ttotal:
            t_eval[-1] = self.ttotal
        t_eval = np.asarray(t_eval, dtype=self.all_dtype)

        ts = t_eval
        # ys = np.ones((ts.shape[0],)+y.shape)+np.nan
        # ys[0] = y0
        # min_h = np.array([999]*ys.shape[-1])

        for i in range(1, len(t_eval)):
            # if i>100:
            #     solver.Global_Minstep = 100
            progress_bar(i - 1, self.nt, bar_length=50)
            solver.t_bound = t_eval[i]
            solver.status = 'running'
            status = None
            while status is None:
                # print(i)
                # i+=1
                message = solver.step()

                if solver.status == 'finished':
                    status = 0
                elif solver.status == 'failed':
                    status = -1
                    break

                # t_old = solver.t_old
                # h_abs = solver.h_abs
                # t = solver.t
                y = solver.y
                # print(t)
            if status != -1:
                result = y
                result = result.reshape(
                    (y.shape[0], 3, self.nsource, self.nzwn))
                lon_new, lat_new = result[0], result[1]
                kx_new, ky_new = result[2], result[3]
                amp_new = result[4]
                nan_indices = np.where(np.abs(lat_new) >= 0.5 * pi)
                lon_new[nan_indices] = np.nan
                lat_new[nan_indices] = np.nan
                kx_new[nan_indices] = np.nan
                ky_new[nan_indices] = np.nan
                amp_new[nan_indices] = np.nan
                ddis = cal_dis(lon_new,lat_new,self.rlon[i - 1],self.rlat[i - 1])
                nan_indices = np.where(np.abs(ddis) >= self.cut_off)
                lon_new[nan_indices] = np.nan
                lat_new[nan_indices] = np.nan
                kx_new[nan_indices] = np.nan
                ky_new[nan_indices] = np.nan
                amp_new[nan_indices] = np.nan

                # print((np.isnan(lon_new)).all())
                if (np.isnan(lon_new)).all() or (
                        np.abs(lat_new) > 0.5 * pi).all():
                    break
                res = self.bs.cal_bs_mercator_point(
                    lon_new.reshape(-1), lat_new.reshape(-1), mode='numpy')

                (fmu, fmv, fmux, fmuy, fmvx, fmvy,
                    fmqx, fmqy, *_) = res
                ug_new, vg_new = cal_ugvg(fmu.reshape(result[0].shape),
                                          fmv.reshape(result[0].shape),
                                          fmqx.reshape(result[0].shape),
                                          fmqy.reshape(result[0].shape),
                                          kx_new, ky_new, mode='extent')
                # print(ug_new,vg_new)

                self.rlon[i, :, :, :] = lon_new
                # print(y[0][0,0,0]/pi*180)
                self.rlat[i, :, :, :] = lat_new
                # print(y[1][0,0,0]/pi*180)
                self.rzwn[i, :, :, :] = kx_new
                self.rmwn[i, :, :, :] = ky_new
                self.ramp[i, :, :, :] = amp_new
                self.rug[i, :, :, :] = ug_new
                self.rvg[i, :, :, :] = vg_new
                solver.y = np.array([
                    self.rlon[i, :, :, :],
                    self.rlat[i, :, :, :],
                    self.rzwn[i, :, :, :],
                    self.rmwn[i, :, :, :],
                    self.ramp[i, :, :, :],
                    # self.rug[0, :, :, :],
                    # self.rvg[0, :, :, :]
                ], dtype=self.all_dtype).reshape((n_vars, -1))
            else:
                break

    def core_ray_run(self, mode='original'):
        if mode == 'original':
            self.core_ray_run_original()
        elif mode == 'numpy':
            self.core_ray_run_numpy()
        elif mode == 'numpy_rk45':
            self.core_ray_run_rk45()

    def ray_run(self, mode='original', inte_method='', root_method='Fortran',
                debug=False, debug_file=None,
                ):
        self.ray_initial(mode=mode, root_method=root_method)
        if debug and not (debug_file is None):
            try:
                self.load_init_from_precal_nc(debug_file)
                print('using init from Fortran for debug')
            except BaseException:
                pass
        if inte_method == 'rk45':
            mode = mode + '_rk45'
        else:
            pass
        self.core_ray_run(mode=mode)

        # 若提前停止，剩余时间步的数组值保持为初始化的 undef（已预填充）
        # end for

    def output(self, ncfile):
        """将射线追踪结果输出到 NetCDF 文件。"""
        out_type = self.all_dtype
        with Dataset(ncfile, 'w') as ds:
            # 定义维度
            ds.createDimension('zwn', self.nzwn)
            ds.createDimension('source', self.nsource)
            ds.createDimension('root', 3)
            ds.createDimension('time', self.nt)
            # 定义坐标变量并赋值
            ds.createVariable('zwn', out_type, ('zwn',))[:] = self.zwn
            ds.createVariable('source_index', 'i4', ('source',))[
                :] = np.arange(self.nsource)
            ds.createVariable('time_index', 'i4', ('time',))[
                :] = np.arange(self.nt)
            # 定义输出变量
            rlon_var = ds.createVariable(
                'rlon', out_type, ('time', 'root', 'source', 'zwn'))
            rlat_var = ds.createVariable(
                'rlat', out_type, ('time', 'root', 'source', 'zwn'))
            rzwn_var = ds.createVariable(
                'rzwn', out_type, ('time', 'root', 'source', 'zwn'))
            rmwn_var = ds.createVariable(
                'rmwn', out_type, ('time', 'root', 'source', 'zwn'))
            ramp_var = ds.createVariable(
                'ramp', out_type, ('time', 'root', 'source', 'zwn'))
            rug_var = ds.createVariable(
                'rug', out_type, ('time', 'root', 'source', 'zwn'))
            rvg_var = ds.createVariable(
                'rvg', out_type, ('time', 'root', 'source', 'zwn'))
            # 写入数据（经纬度转换为度）
            rlon_var[:] = self.rlon * rad2deg
            rlat_var[:] = self.rlat * rad2deg
            rzwn_var[:] = self.rzwn
            rmwn_var[:] = self.rmwn
            ramp_var[:] = self.ramp
            rug_var[:] = self.rug   # 群速度 (m/s)
            rvg_var[:] = self.rvg
            # 可选：添加属性说明单位
            rlon_var.units = 'degrees'
            rlat_var.units = 'degrees'
            rzwn_var.units = 'rad_per_meter*Rearth'  # 或 'a^-1'
            rug_var.units = 'm s-1'
            rvg_var.units = 'm s-1'

    def clean(self):
        """释放 WR 对象资源 (包含BS)。"""
        self.bs.clean()
        attrs = [
            "rlon",
            "rlat",
            "rzwn",
            "rmwn",
            "ramp",
            "rug",
            "rvg",
            "source_lon",
            "source_lat",
            "zwn"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
