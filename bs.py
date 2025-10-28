# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 22:47:13 2025

@author: 杨艺楠
"""

# bs.py - 基本流场模块
import numpy as np
from constants import pi, rearth, omega, undef, delt
import netCDF4 as nc
from netCDF4 import Dataset
from interpolation import batch_linint2_metpy
from scipy.ndimage import convolve
import numba as nb
from interpolation import linint2_point
import sys
import os
_cmplx_so_exist_ = True
try:
    path_to_so = 'E:\1_py\0715wr_py\cmplx_roots_sg.cpython-38-x86_64-linux-gnu.so' # test
    if path_to_so not in sys.path:
        sys.path.insert(0, path_to_so)
    import cmplx_roots_sg
    _cmplx_so_exist_ = True
    # print(">>> 成功导入 cmplx_roots_sg ✅")
except Exception as e:
    # print(">>> 导入失败 ❌:", e)
    _cmplx_so_exist_ = False

nb_para_dic = {
    'nopython': True,
    # 'fastmath':True,
    'cache': True,
}


@nb.jit(['c16[:](c16[:])'], **nb_para_dic)  # ,'c8[:](c8[:])'
def roots_numba(p):
    return np.roots(p)

if _cmplx_so_exist_:
    def roots_Fortran(p):
        '''
        使用f2py引用原代码中的cmplx_roots_sg
        使用方法为：
        首先 export CFLAGS="-std=c99"
        然后 python -m numpy.f2py -c cmplx_roots_sg.f90 -m cmplx_roots_sg
        之后可以通过import cmplx_roots_sg使用
        使用print(cmplx_roots_sg.__doc__)查看接口
        cmplx_roots_gen(roots,poly,polish_roots_after,use_roots_as_starting_points,degree=shape(roots, 0))
        roots : np.array,dtype = 'complex128',shape = degree
        poly : np.array, dtype = 'complex128',shape = degree+1
        按原代码,polish_roots_after,use_roots_as_starting_points = True, False
        '''
        p = p.astype('complex128')
        p = p[::-1]
        roots = np.ones((len(p) - 1), dtype='complex128')
        cmplx_roots_sg.cmplx_roots_gen(roots, p, True, False)
        return roots

def roots_(p, root_method='Fortran'):
    if root_method == 'Fortran':
        return roots_Fortran(p)
    elif root_method == 'numpy':
        return roots_numba(p)


class BS:
    def __init__(self, nlon, nlat, read_dtype='float32', cal_dtype='float64'):
        self.all_dtype = read_dtype
        self.all_dtype_ = cal_dtype
        # pi, rearth, omega, undef, delt = np.array([pi, rearth, omega, undef, delt],dtype=cal_dtype)
        self.nlon = nlon
        self.nlat = nlat
        shape = (nlon, nlat)
        self.dx = np.array([2.0 * pi / self.nlon], dtype=self.all_dtype_)
        self.dy = np.array([pi / (self.nlat - 1)], dtype=self.all_dtype_)
        # 风场和涡度
        self.u = np.zeros(shape, dtype=self.all_dtype)
        self.v = np.zeros(shape, dtype=self.all_dtype)
        self.q = np.zeros(shape, dtype=self.all_dtype_)
        self.lat = np.zeros(nlat, dtype=self.all_dtype_)
        self.lon = np.zeros(nlon, dtype=self.all_dtype_)
        # 一阶导数
        self.ux = np.zeros(shape, dtype=self.all_dtype_)
        self.vx = np.zeros(shape, dtype=self.all_dtype_)
        self.qx = np.zeros(shape, dtype=self.all_dtype_)
        self.uy = np.zeros(shape, dtype=self.all_dtype_)
        self.vy = np.zeros(shape, dtype=self.all_dtype_)
        self.qy = np.zeros(shape, dtype=self.all_dtype_)
        # 二阶导数
        self.uxx = np.zeros(shape, dtype=self.all_dtype_)
        self.vxx = np.zeros(shape, dtype=self.all_dtype_)
        self.qxx = np.zeros(shape, dtype=self.all_dtype_)
        self.uyy = np.zeros(shape, dtype=self.all_dtype_)
        self.vyy = np.zeros(shape, dtype=self.all_dtype_)
        self.qyy = np.zeros(shape, dtype=self.all_dtype_)
        self.uxy = np.zeros(shape, dtype=self.all_dtype_)
        self.vxy = np.zeros(shape, dtype=self.all_dtype_)
        self.qxy = np.zeros(shape, dtype=self.all_dtype_)
        self.qyx = np.zeros(shape, dtype=self.all_dtype_)
        # 三阶导数
        self.qxxx = np.zeros(shape, dtype=self.all_dtype_)
        self.qxxy = np.zeros(shape, dtype=self.all_dtype_)
        self.qxyy = np.zeros(shape, dtype=self.all_dtype_)
        self.qyyy = np.zeros(shape, dtype=self.all_dtype_)
        self.qyxx = np.zeros(shape, dtype=self.all_dtype_)
        self.qyyx = np.zeros(shape, dtype=self.all_dtype_)
        # beta_m 和 K_S 场
        self.betam = np.zeros(shape, dtype=self.all_dtype_)
        self.KS = np.zeros(shape, dtype=self.all_dtype_)

    # 外部调用，统一接口
    def getlon(self):
        return self.lon

    def getlat(self):
        return self.lat

    def gradient_x(self, f_):
        """计算沿经度方向的一阶导数 df/d(lambda)。对经度环向做周期处理。"""
        f = f_.astype(self.all_dtype_)
        fx = np.empty_like(f, dtype=self.all_dtype_)
        # 中部点用中心差分
        fx[1:-1, :] = (f[2::, :] - f[0:-2, :]) / (2.0 * self.dx)
        # 周期边界: 首列和末列
        fx[0, :] = (f[1, :] - f[-1, :]) / (2.0 * self.dx)
        fx[-1, :] = (f[0, :] - f[-2, :]) / (2.0 * self.dx)
        # print('fx_dtype: ',fx.dtype)
        return fx  # .astype(self.all_dtype_)

    def gradient_y(self, f_):
        """计算沿纬度方向的一阶导数 df/d(phi)。纬度方向非周期，边界用单侧差分。"""
        f = f_.astype(self.all_dtype_)
        fy = np.empty_like(f, dtype=self.all_dtype_)
        # 中部点中心差分
        fy[:, 1:-1] = (f[:, 2::] - f[:, 0:-2]) / (2.0 * self.dy)
        # 南北边界: 用前向/后向差分近似
        fy[:, 0] = (f[:, 1] - f[:, 0]) / (self.dy)
        fy[:, -1] = (f[:, -1] - f[:, -2]) / (self.dy)
        return fy  # .astype(self.all_dtype_)

    def gradient_xx(self, f_):
        """计算沿经度方向的二阶导数 d^2 f/d(lambda)^2。经度方向视为环状。"""
        f = f_.astype(self.all_dtype_)
        fxx = np.empty_like(f, dtype=self.all_dtype_)
        # 中部点中心差分近似二阶导
        fxx[1:-1, :] = (f[2::, :] - 2.0 * f[1:-1, :] +
                        f[0:-2, :]) / (self.dx**2)
        # 边界点
        fxx[0, :] = (f[1, :] - 2.0 * f[0, :] + f[-1, :]) / (self.dx**2)
        fxx[-1, :] = (f[0, :] - 2.0 * f[-1, :] + f[-2, :]) / (self.dx**2)
        return fxx  # .astype(self.all_dtype_)

    def gradient_yy(self, f_):
        """计算沿纬度方向的二阶导数 d^2 f/d(phi)^2。纬度边界处结果复制邻点。"""
        f = f_.astype(self.all_dtype_)
        fyy = np.empty_like(f, dtype=self.all_dtype_)
        # 中部点中心差分
        fyy[:, 1:-1] = (f[:, 2::] - 2.0 * f[:, 1:-1] +
                        f[:, 0:-2]) / (self.dy**2)
        # 边界: 复制相邻点的值（假设边界导数与邻格相同）
        fyy[:, 0] = fyy[:, 1]
        fyy[:, -1] = fyy[:, -2]
        return fyy

    def gradient_xy(self, f):
        """计算先沿经度后沿纬度的二阶混合导数 d^2 f/(dλ dφ)。"""
        # nx, ny = self.nlon, self.nlat
        fxy = np.empty_like(f, dtype=self.all_dtype_)
        # 中心区域使用四点差分计算混合偏导
        # for i in range(1, nx-1):
        #     for j in range(1, ny-1):
        #         fxy[i, j] = (f[i+1, j+1] - f[i+1, j-1] - f[i-1, j+1] + f[i-1, j-1]) / (4.0 * self.dx * self.dy)
        fxy[1:-1,
            1:-1] = (f[2::,
                       2::] - f[2::,
                                0:-2] - f[0:-2,
                                          2::] + f[0:-2,
                                                   0:-2]) / (4.0 * self.dx * self.dy)

        fxy[1:-1, 0] = fxy[1:-1, 1]
        fxy[1:-1, -1] = fxy[1:-1, -2]

        fxy[0, 1:-1] = (f[1, 2::] - f[1, 0:-2] - f[-1, 2::] +
                        f[-1, 0:-2]) / (4 * self.dx * self.dy)
        fxy[-1, 1:-1] = (f[0, 2::] - f[0, 0:-2] - f[-2, 2::] +
                         f[-2, 0:-2]) / (4 * self.dx * self.dy)

        fxy[0, 0] = fxy[0, 1]
        fxy[0, -1] = fxy[0, -2]
        fxy[-1, 0] = fxy[-1, 1]
        fxy[-1, -1] = fxy[-1, -2]
        return fxy  # .astype(self.all_dtype_)

    def gradient_yx(self, f):
        """计算先沿纬度后沿经度的混合导数 (与 gradient_xy 相同)。"""
        # 由于连续偏导次序可交换，这里直接调用 gradient_xy 实现
        return self.gradient_xy(f)

    def loadbs_ncfile(self, ncfile):
        '''
        如果文件中存在lon和lat，使用它们做为坐标
        如果不存在lon或lat，使用维度来构建，默认-1维为lon，-2维为lat，并给出warning
        默认为0E->360E和90S->90N
        '''
        ds = nc.Dataset(ncfile)

        temp_u = np.array(ds.variables['u'][:], dtype=self.all_dtype)
        temp_v = np.array(ds.variables['v'][:], dtype=self.all_dtype)

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
                self.lat[:] = (
                    lat_data *
                    pi /
                    180).astype(
                    self.all_dtype_)  # 转为弧度
                break
        for name in lon_candidates:
            if name in ds.variables:
                lon_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                self.lon[:] = (lon_data * pi / 180).astype(self.all_dtype_)
                break

        # 如果找不到就构造规则网格
        if lat_data is None:
            self.lat = - pi * 0.5 + np.arange(self.nlat) * self.dy
        if lon_data is None:
            self.lon = np.arange(self.nlon) * self.dx
        # 确保纬度递增
        self.u = np.transpose(temp_u, (1, 0))
        # python 的netCDF4读取出来顺序是与fortran相反的,这里添加了转置
        self.v = np.transpose(temp_v, (1, 0))
        if (lat_data is None) or (lon_data is None):
            message = '###WARNING: lon and lat not found. Make sure your lats are from 90S to 90N and lons are from 0E to 360E###'
            print(message)
        elif not (lat_data is None):
            if lat_data[0] > lat_data[-1]:
                lat_data = lat_data[::-1]
                self.u = np.transpose(temp_u[::-1, :], (1, 0))
                # python 的netCDF4读取出来顺序是与fortran相反的,这里添加了转置
                self.v = np.transpose(temp_v[::-1, :], (1, 0))

        # import matplotlib.pyplot as plt
        # plt.contourf(ds.variables['v'][:])
        # plt.show()

        ds.close()

    def calc_absolute_vorticity(self):
        # u_cos = np.empty_like(self.u)
        # for j in range(ny):
        #     u_cos[:, j] = self.u[:, j] * np.cos(self.lat[j])
        # u1 = self.u.astype(self.all_dtype_)
        v1 = self.v.astype(self.all_dtype_)
        u_cos = (self.u * np.cos(self.lat[None, :])).astype(self.all_dtype_)
        u_cos_y = self.gradient_y(u_cos)
        v_x = self.gradient_x(v1)
        # for j in range(1, ny-1):
        #     self.q[:, j] = (v_x[:, j] - u_cos_y[:, j]) / np.cos(self.lat[j]) \
        #                     + 2.0 * omega * np.sin(self.lat[j]) * rearth
        self.q[:, 1:-1] = (v_x[:, 1:-1] - u_cos_y[:, 1:-1]) / np.cos(self.lat[1:-1])[None, :] \
            + 2.0 * omega * np.sin(self.lat[1:-1])[None, :] * rearth
        self.q[:, 0] = self.q[:, 1]
        self.q[:, -1] = self.q[:, -2]

    def clean(self):
        # 变量缓存清空器
        attrs = ["u", "v", "q", "ux", "vx", "qx", "uy", "vy", "qy", "uxx", "vxx", "qxx",
                 "uyy", "vyy", "qyy", "uxy", "vxy", "qxy", "qyx", "qxxx", "qxxy", "qxyy",
                 "qyyy", "qyxx", "qyyx", "betam", "KS"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
                
    # 20250718已验证：Python和Fortran的smth9实现计算方式、权重定义、边界处理、更新逻辑完全一致，无计算差异。
    def smth9(self, field, p=0.5, q=0.25):
        """
        九点加权平滑器，高效 NumPy 卷积实现。
        p: 十字方向（上下左右）权重系数
        q: 角点方向（四角）权重系数
        """
        kernel = np.array([
            [q / 4, p / 4, q / 4],
            [p / 4, -(p + q), p / 4],
            [q / 4, p / 4, q / 4]
        ])
        smooth = field
        smooth[1:-2, 1:-2] = smooth[1:-2, 1:-2] + \
            convolve(field, kernel, mode='constant', cval=0.0)[1:-2, 1:-2]
        return smooth
    # def smth9(self,field, p=0.5, q=0.25):
    #     '''
    #     original method
    #     '''
    #     m, n = field.shape
    #     f_sm = field.copy()
    #     for j in range(1, n-2):
    #         for i in range(1, m-2):
    #             f_sm[i, j] = f_sm[i, j] + p/4.0 * (f_sm[i-1,j] + f_sm[i+1,j] + f_sm[i,j-1] + f_sm[i,j+1] - 4*f_sm[i,j]) \
    #                                         + q/4.0 * (f_sm[i-1,j-1] + f_sm[i-1,j+1] + f_sm[i+1,j-1] + f_sm[i+1,j+1] - 4*f_sm[i,j])
    #     return f_sm

    def ready(self, xcyclic=False):
        self.xcyclic = xcyclic
        nx, ny = self.nlon, self.nlat
        self.calc_absolute_vorticity()
        self.ux = self.gradient_x(self.u)
        self.uy = self.gradient_y(self.u)
        self.vx = self.gradient_x(self.v)
        self.vy = self.gradient_y(self.v)
        self.qx = self.gradient_x(self.q)
        self.qy = self.gradient_y(self.q)
        self.uxx = self.gradient_xx(self.u)
        self.uyy = self.gradient_yy(self.u)
        self.vxx = self.gradient_xx(self.v)
        self.vyy = self.gradient_yy(self.v)
        self.qxx = self.gradient_xx(self.q)
        self.qyy = self.gradient_yy(self.q)
        self.uxy = self.gradient_xy(self.u)
        self.vxy = self.gradient_xy(self.v)
        self.qxy = self.gradient_xy(self.q)
        self.qyx = self.qxy.copy()
        self.qxxx = self.gradient_x(self.qxx)
        self.qxxy = self.gradient_y(self.qxx)
        self.qxyy = self.gradient_y(self.qxy)
        self.qyyy = self.gradient_y(self.qyy)
        self.qyxx = self.gradient_x(self.qxy)
        self.qyyx = self.gradient_x(self.qyy)

        self.qxx = self.smth9(self.qxx)
        self.qyy = self.smth9(self.qyy)
        self.qxy = self.smth9(self.qxy)

        self.fields = np.stack([
            self.u,
            self.v,
            self.ux,
            self.uy,
            self.vx,
            self.vy,
            self.qx,
            self.qy,
            self.qxx,
            self.qxy,
            self.qyx,
            self.qyy,
            self.qxxx,
            self.qxxy,
            self.qxyy,
            self.qyyy,
            self.qyxx,
            self.qyyx
        ], axis=-1)
        self.fields = self.fields.astype(self.all_dtype_)
        if self.xcyclic:
            self.fields = np.concatenate(
                [self.fields, self.fields[0:1, :, :]], axis=0)
        # for j in range(1, ny-1):
        #     for i in range(nx):
        #         self.betam[i, j] = (2 * omega * (np.cos(self.lat[j])**2) \
        #             + (-np.cos(self.lat[j]) * self.uyy[i, j] \
        #                + np.sin(self.lat[j]) * self.uy[i, j] \
        #                + self.u[i, j] / np.cos(self.lat[j])) / rearth) / rearth
        self.betam[:, 1:-1] = (2 * omega * (np.cos(self.lat[None, 1:-1])**2)
                               + (-np.cos(self.lat[None, 1:-1]) * self.uyy[:, 1:-1]
                                  + np.sin(self.lat[None, 1:-1]
                                           ) * self.uy[:, 1:-1]
                                  + self.u[:, 1:-1] / np.cos(self.lat[None, 1:-1])) / rearth) / rearth
        self.betam[:, 0] = undef
        self.betam[:, ny - 1] = undef

        # for j in range(1, ny-1):
        #     for i in range(nx):
        #         if self.betam[i, j] > 0.0 and self.u[i, j] > 0.0:
        #             self.KS[i, j] = np.sqrt(self.betam[i, j] * np.cos(self.lat[j]) / self.u[i, j]) * rearth
        #         else:
        #             self.KS[i, j] = undef

        mask_betam = np.ones(self.betam.shape)
        mask_u = np.ones(self.u.shape)
        mask_betam[self.betam <= 0] = 0
        mask_u[self.u <= 0] = 0
        self.KS[:,
                1:-1] = np.sqrt(self.betam[:,
                                           1:-1] * np.cos(self.lat[None,
                                                                   1:-1]) / self.u[:,
                                                                                   1:-1]) * rearth
        self.KS = self.KS * mask_betam * mask_u  # +(1-mask_u*mask_betam)*undef
        self.KS[mask_u * mask_betam == 0] = undef

        self.KS[:, 0] = undef
        self.KS[:, ny - 1] = undef

    def read_from_precaled_nc(self, ncfile):
        '''
        使用由fortran计算的基本场，用于debug
        '''
        ds = nc.Dataset(ncfile)

        attrs = ["lat", "lon", "u", "v", "q", "ux", "vx", "qx", "uy", "vy", "qy", "uxx", "vxx", "qxx",
                 "uyy", "vyy", "qyy", "uxy", "vxy", "qxy", "qyx", "qxxx", "qxxy", "qxyy",
                 "qyyy", "qyxx", "qyyx", "betam", "KS"]
        print('USING BASEFILE CALCULATED BY FORTRAN FOR DEBUG')
        for name in attrs:
            if name in ['lon', 'lat']:
                setattr(self, name, ds.variables[name][:] / 180 * pi)
            elif name in ['qyx']:
                setattr(
                    self, name, np.transpose(
                        ds.variables['qxy'][:], (1, 0)))
            else:
                # print(name,ds.variables[name.lower()][:].shape,ds.variables[name.lower()][:].dtype)
                setattr(self, name, np.transpose(
                    ds.variables[name.lower()][:], (1, 0)))
        self.fields = np.stack([
            self.u,
            self.v,
            self.ux,
            self.uy,
            self.vx,
            self.vy,
            self.qx,
            self.qy,
            self.qxx,
            self.qxy,
            self.qyx,
            self.qyy,
            self.qxxx,
            self.qxxy,
            self.qxyy,
            self.qyyy,
            self.qyxx,
            self.qyyx
        ], axis=-1)
        self.fields = self.fields.astype(self.all_dtype_)
        if self.xcyclic:
            self.fields = np.concatenate(
                [self.fields, self.fields[0:1, :, :]], axis=0)
        ds.close()

    def terminate_ray(self, rlon, rlat, ramp, rug, rvg,
                      rzwn, rmwn, iz, isrc, ir, it):
        for arr in [rlon, rlat, ramp, rug, rvg, rzwn, rmwn]:
            arr[it + 1:, ir, isrc, iz] = np.nan

    def output(self, ncfile):
        """
        将基本流场输出为 NetCDF4 文件，支持压缩、变量自动写入。
        是bs_file哦，nc_file在wr.py输出
        """
        output_type = self.all_dtype_
        with Dataset(ncfile, 'w', format='NETCDF4') as ds:
            # 创建维度
            ds.createDimension('lon', self.nlon)
            ds.createDimension('lat', self.nlat)

            # 写入经纬度
            for name, data in zip(
                    ['lon', 'lat'], [self.getlon(), self.getlat()]):
                var = ds.createVariable(name, output_type, (name,))
                var[:] = data
                var.units = 'degrees_east' if name == 'lon' else 'degrees_north'

            # 统一写入二维变量（可轻松扩展）

            field_map = {
                'u': (self.u, 'm/s'),
                'v': (self.v, 'm/s'),
                'q': (self.q, '1/s'),
                'ux': (self.ux, 'None'),
                'uxx': (self.uxx, 'None'),
                'uy': (self.uy, 'None'),
                'vx': (self.vx, 'None'),
                'vxx': (self.vxx, 'None'),
                'vy': (self.vy, 'None'),
                'qx': (self.qx, 'None'),
                'qy': (self.qy, 'None'),
                'qxx': (self.qxx, 'None'),
                'qxy': (self.qxy, 'None'),
                'qyx': (self.qyx, 'None'),
                'qyy': (self.qyy, 'None'),
                'qxxx': (self.qxxx, 'None'),
                'qxxy': (self.qxxy, 'None'),
                'qxyy': (self.qxyy, 'None'),
                'qyyy': (self.qyyy, 'None'),
                'qyxx': (self.qyxx, 'None'),
                'qyyx': (self.qyyx, 'None'),
                'betam': (self.betam, '1/(m·s)'),
                'KS': (self.KS, '1/m'),
            }

            for name, (data, unit) in field_map.items():
                var = ds.createVariable(
                    name, output_type, ('lon', 'lat'), zlib=True, complevel=4)
                var[:, :] = data
                var.units = unit

    def cal_bs_mercator_point(self, lon, lat, mode='original'):
        """
        在经纬度点 (lon, lat) 上插值计算基本流场及其导数，并转换到 Mercator 投影坐标系下。
        返回在该点的一系列物理量。
        """
        from numpy import pi, cos, sin, tan
        lon = lon % (2 * pi)
        wrapX = True
        if mode == 'original':
            """
            在经纬度点 (lon, lat) 上插值计算基本流场及其导数，并转换到 Mercator 投影坐标系下。
            返回一系列在该点的物理量:
            fmu, fmv   - Mercator投影下的风 (即 u/cosφ, v/cosφ)
            fmux,fmuy, fmvx,fmvy - 对应 fmu,fmv 的经纬偏导数
            fmqx,fmqy            - Mercator 投影下绝对涡度 q 的经纬导数
            fmqxx,fmqxy,fmqyx,fmqyy - q 的二阶偏导数 (在 Mercator 坐标)
            fmqxxx,...fmqyyx     - q 的三阶偏导数 (Mercator 坐标)
            """
            # 确保 lon 在 [0, 2π) 范围
            lon = lon % (2 * pi)
            # 确保 lat 在 [-pi/2, pi/2] 范围
            if lat < -0.5 * pi or lat > 0.5 * pi:
                print("ERROR in cal_bs_mercator_point: latitude out of range!")
                return None
            # 调用双线性插值获取原始变量在该点的值 (使用缺省undef标记)
            mm, nn = self.nlon, self.nlat
            # 因为 lon轴是环向周期的:
            wrapX = True
            # 插值基本场及导数
            # import matplotlib.pyplot as plt
            # plt.contourf(self.u)
            # plt.show()
            fu = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.u,
                wrapX,
                lon,
                lat,
                undef)
            # print(fu)
            fv = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.v,
                wrapX,
                lon,
                lat,
                undef)
            fux = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.ux,
                wrapX,
                lon,
                lat,
                undef)
            fuy = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.uy,
                wrapX,
                lon,
                lat,
                undef)
            fvx = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.vx,
                wrapX,
                lon,
                lat,
                undef)
            fvy = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.vy,
                wrapX,
                lon,
                lat,
                undef)
            fqx = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qx,
                wrapX,
                lon,
                lat,
                undef)
            fqy = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qy,
                wrapX,
                lon,
                lat,
                undef)
            fqxx = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qxx,
                wrapX,
                lon,
                lat,
                undef)
            fqxy = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qxy,
                wrapX,
                lon,
                lat,
                undef)
            fqyx = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qyx,
                wrapX,
                lon,
                lat,
                undef)
            fqyy = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qyy,
                wrapX,
                lon,
                lat,
                undef)
            fqxxx = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qxxx,
                wrapX,
                lon,
                lat,
                undef)
            fqxxy = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qxxy,
                wrapX,
                lon,
                lat,
                undef)
            fqxyy = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qxyy,
                wrapX,
                lon,
                lat,
                undef)
            fqyyy = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qyyy,
                wrapX,
                lon,
                lat,
                undef)
            fqyxx = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qyxx,
                wrapX,
                lon,
                lat,
                undef)
            fqyyx = linint2_point(
                mm,
                self.lon,
                nn,
                self.lat,
                self.qyyx,
                wrapX,
                lon,
                lat,
                undef)
            # 将原始场转换到 Mercator 坐标:
            # 注意 Mercator 纬度 φ 在 [-π/2, π/2] 范围
            cos_phi = np.cos(lat)
            sin_phi = np.sin(lat)
            if abs(cos_phi) > 0.0175:  # cos(phi) 不太小 (约<89度)
                # 转换涡度导数到 Mercator (经度方向变化不变，纬度方向乘 cosφ，因为 d/dy_mer = cosφ *
                # d/dφ)
                fmqxx = fqxx
                fmqyx = fqxy * cos_phi   # q_{yλ} = q_{λφ} * cosφ
                fmqxy = fmqyx            # 对称性
                fmqyy = (fqyy * cos_phi - fqy * sin_phi) * cos_phi
                # 三阶导数
                fmqxxx = fqxxx
                fmqxxy = fqxxy * cos_phi
                fmqxyy = (fqxyy * cos_phi - fqxy * sin_phi) * cos_phi
                fmqyyy = fqyyy
                fmqyxx = fqyxx * cos_phi
                fmqyyx = (fqyyx * cos_phi - fqxy * sin_phi) * cos_phi
                # 转换一阶导数:
                fmqx = fqx               # 经度方向导数不变
                fmqy = fqy * cos_phi     # 纬度导数乘 cosφ
                # 转换风场导数:
                fmux = fux / cos_phi
                fmvx = fvx / cos_phi
                # 注意: 按公式应除以cosφ，但Fortran实现未除，为保持一致，此处不再额外除cosφ
                fmuy = fuy + np.tan(lat) * fu
                fmvy = fvy + np.tan(lat) * fv   # 同上
                # 转换风场本身:
                fmu = fu / cos_phi
                fmv = fv / cos_phi
            else:
                # 非法或近极值，直接返回0或缺测
                print('cos(phi) close to zero')
                fmu = 0.0
                fmv = 0.0
                fmux = 0.0
                fmuy = 0.0
                fmvx = 0.0
                fmvy = 0.0
                fmqx = 0.0
                fmqy = 0.0
                fmqxx = 0.0
                fmqxy = 0.0
                fmqyx = 0.0
                fmqyy = 0.0
                fmqxxx = 0.0
                fmqxxy = 0.0
                fmqxyy = 0.0
                fmqyyy = 0.0
                fmqyxx = 0.0
                fmqyyx = 0.0
            return (fmu, fmv, fmux, fmuy, fmvx, fmvy,
                    fmqx, fmqy, fmqxx, fmqxy, fmqyx, fmqyy,
                    fmqxxx, fmqxxy, fmqxyy, fmqyyy, fmqyxx, fmqyyx)

        elif mode == 'numpy':
            '''
            add at 20250407 23:27
            shapes: 1-d array
            '''

            in_range_indices = np.where(np.abs(lat) <= 0.5 * pi)[0]

            xo = lon  # np.array([lon])
            yo = lat  # np.array([lat])
            # import matplotlib.pyplot as plt
            # plt.contourf(self.u.T)
            # plt.show()
            # fields = np.stack([
            #     self.u,
            #     self.v,
            #     self.ux,
            #     self.uy,
            #     self.vx,
            #     self.vy,
            #     self.qx,
            #     self.qy,
            #     self.qxx,
            #     self.qxy,
            #     self.qyx,
            #     self.qyy,
            #     self.qxxx,
            #     self.qxxy,
            #     self.qxyy,
            #     self.qyyy,
            #     self.qyxx,
            #     self.qyyx
            # ],axis=-1)
            # print(self.qy[0,520])
            # print(self.qy[-1,520])
            # import matplotlib.pyplot as plt
            # mask = np.ones(self.qy.shape)-1
            # mask[self.qy==0]=1
            # plt.contourf(np.ma.array(self.qy,mask=mask))
            # plt.show()

            interp_fields_ = np.ones(
                (self.fields.shape[-1], len(lat))) * np.nan
            interp_fields = batch_linint2_metpy(
                self.lon,
                self.lat,
                self.fields,
                xo[in_range_indices],
                yo[in_range_indices],
                fo_missing=undef,
                xcyclic=wrapX,
                mode='numpy',
                all_dtype=self.all_dtype_)
            interp_fields_[
                :, in_range_indices] = np.transpose(
                interp_fields, (1, 0))
            (fu,
             fv,
             fux,
             fuy,
             fvx,
             fvy,
             fqx,
             fqy,
             fqxx,
             fqxy,
             fqyx,
             fqyy,
             fqxxx,
             fqxxy,
             fqxyy,
             fqyyy,
             fqyxx,
             fqyyx) = interp_fields_
            # print('fu',fu,'fv',fv)
            cos_phi = cos(lat)
            sin_phi = sin(lat)
            mask = np.ones(cos_phi.shape, dtype=self.all_dtype_)
            mask[np.abs(cos_phi) <= 0.0175] = 0
            cos_phi = cos_phi * mask + (1 - mask) * 1e-6

            fmqxx = fqxx * mask
            fmqyx = fqxy * cos_phi * mask
            fmqxy = fmqyx * mask
            fmqyy = (fqyy * cos_phi - fqy * sin_phi) * cos_phi * mask

            fmqxxx = fqxxx * mask
            fmqxxy = fqxxy * cos_phi * mask
            fmqxyy = (fqxyy * cos_phi - fqxy * sin_phi) * cos_phi * mask
            fmqyyy = fqyyy * mask
            fmqyxx = fqyxx * cos_phi * mask
            fmqyyx = (fqyyx * cos_phi - fqxy * sin_phi) * cos_phi * mask

            fmqx = fqx * mask
            fmqy = fqy * cos_phi * mask

            fmux = fux / cos_phi * mask
            fmvx = fvx / cos_phi * mask
            fmuy = (fuy + tan(lat) * fu) * mask
            fmvy = (fvy + tan(lat) * fv) * mask

            fmu = fu / cos_phi * mask
            fmv = fv / cos_phi * mask

        return np.array([fmu, fmv, fmux, fmuy, fmvx, fmvy,
                         fmqx, fmqy, fmqxx, fmqxy, fmqyx, fmqyy,
                         fmqxxx, fmqxxy, fmqxyy, fmqyyy, fmqyxx, fmqyyx], dtype=self.all_dtype_)


# 多项式求根工具 (利用 numpy.roots 实现 Jenkins-Traub 算法求复根，然后筛选实根)

def cal_ky_original(fu, fv, fqx, fqy, freq, zwn):
    """
    计算给定背景场参数下的传播的 meridional 波数 m (最多3个解)。
    对应 Fortran 的 cal_ky 子程序。
    输入:
      fu, fv   - 在 Mercator 上的 u/cosφ, v/cosφ (单位: m/s)
      fqx, fqy - 绝对涡度 q 关于经度和纬度的梯度 (Mercator坐标, qx 已乘Rearth, qy已乘Rearth)
      freq     - 波的频率 (rad/s)
      zwn      - 初始的无量纲 zonal wave number (k * Rearth)
    返回:
      mwn_list[0..2] - 长度3的 meridional 波数数组 (未找到用 undef 填充)
      len(real_roots)        - 找到的实根个数
    """
    if zwn == 0.0:
        return [undef] * 3, 0

    # 构建多项式系数（按 m^3 -> m^0 排列）
    ps = freq / zwn * rearth
    coeff = [
        (zwn**3) * (fu - ps - (fqy / zwn**2)),
        (zwn**2) * fv + fqx,
        zwn * (fu - ps),
        fv
    ]
    coeff = coeff[::-1]  # 转换为 [常数项, ..., 最高次项]

    # 自动降阶（忽略几乎为零的高阶项）
    while coeff and abs(coeff[-1]) < delt:
        coeff.pop()
    deg = len(coeff) - 1

    if deg < 1:
        return [undef] * 3, 0

    coeff = [x.item() if isinstance(x, np.ndarray) else x for x in coeff]
    # 求实根
    roots = np.roots(coeff)
    real_roots = [
        r.real for r in roots if abs(
            r.imag) < delt and abs(
            r.real) < 100.0]

    # 排序策略：优先非负、再按数值大小排序
    real_roots.sort(key=lambda x: (x < 0, abs(x)))

    # 补齐为3个根
    mwn_list = real_roots[:3] + [undef] * (3 - len(real_roots))
    return mwn_list, len(real_roots)


def change_roots_order(mwn, deg):
    '''
    mwn: roots with undef
    deg: root_num
    '''
    if deg == 3:

        bestidx = 1
        for i in range(1, 3):
            if mwn[i] >= 0. and mwn[i] < mwn[bestidx]:
                mwn[i], mwn[bestidx] = mwn[bestidx], mwn[i]
                bestidx = i

        if mwn[0] < 0:
            mwn[0], mwn[1] = mwn[1], mwn[0]

        if (mwn[1] < 0 and mwn[2] < 0 and mwn[1] <
                mwn[2]) or (mwn[1] > 0 and mwn[2] < 0.):
            mwn[1], mwn[2] = mwn[2], mwn[1]

    elif deg == 2:
        for i in range(3):
            if (not np.isnan(mwn[i])) and mwn[i] > 0:
                mwn[i], mwn[0] = mwn[0], mwn[i]
                break
            else:
                mwn[i], mwn[1] = mwn[1], mwn[i]
                break

    elif deg == 1:
        for i in range(3):
            if (not np.isnan(mwn[i])) and mwn[i] >= 0 and i != 0:
                mwn[i], mwn[0] = mwn[0], mwn[i]
            elif (not np.isnan(mwn[i])) and mwn[i] <= 0. and i != 2:
                mwn[i], mwn[1] = mwn[1], mwn[i]

    for i in range(3):
        if (not np.isnan(mwn[i])) and abs(mwn[i]) > 100.:
            mwn[i] = np.nan
            deg = deg - 1
    return mwn[::-1], deg


def cal_ky_numpy(fu, fv, fqx, fqy, freq, zwn, root_method='Fortran'):
    """
    计算给定背景场参数下的传播的 meridional 波数 m (最多3个解)。
    对应 Fortran 的 cal_ky 子程序。
    输入:
      fu, fv   - 在 Mercator 上的 u/cosφ, v/cosφ (单位: m/s)
      fqx, fqy - 绝对涡度 q 关于经度和纬度的梯度 (Mercator坐标, qx 已乘Rearth, qy已乘Rearth)
      freq     - 波的频率 (rad/s)
      zwn      - 初始的无量纲 zonal wave number (k * Rearth)
    返回:
      mwn_list[0..2] - 长度3的 meridional 波数数组 (未找到用 undef 填充)
      len(real_roots)        - 找到的实根个数

    shape = (points,) + (original.shape) for fu,fv,fqx,fqy
    freq, zwn: float or int
    """
    mwn_list = np.ones((len(fu), 3)) * np.nan
    lens = np.ones((len(fu))) - 1
    if zwn != 0:
        # 构建多项式系数（按 m^3 -> m^0 排列）
        ps = freq / zwn * rearth
        coeff_ = np.stack([
            (zwn**3) * (fu - ps - (fqy / zwn**2)),
            (zwn**2) * fv + fqx,
            zwn * (fu - ps),
            fv
        ], axis=-1)
        # coeff_ = coeff_[:,::-1]  # 转换为 [常数项, ..., 最高次项]
        # print('coeff',coeff_)
        for i in range(coeff_.shape[0]):
            coeff = coeff_[i, :]
            # 自动降阶（忽略几乎为零的高阶项）
            deg = 3
            while deg > 0 and abs(coeff[deg]) == 0:
                deg -= 1

            coeff = coeff[0:deg + 1]

            if deg < 1:
                mwn_list[i, :] = np.array([undef] * 3)
                lens[i] = 0

            # 求实根
            else:
                roots = roots_(coeff[::-1] + 0j, root_method=root_method)
                real_roots = [r.real for r in roots if abs(r.imag) < delt]
                roots_num = len(real_roots)
                mwn = np.array(real_roots[:3] + [undef] * (3 - roots_num))
                mwn, roots_num = change_roots_order(mwn, roots_num)
                # 排序策略：优先非负、再按数值大小排序
                # real_roots.sort(key=lambda x: (x < 0, abs(x)))

                # 补齐为3个根
                mwn_list[i, :] = mwn
                lens[i] = roots_num
    return mwn_list, lens


def cal_ky(fu, fv, fqx, fqy, freq, zwn,iz=0,
           mode='original', root_method='Fortran'):
    if mode == 'original':
        return cal_ky_original(fu, fv, fqx, fqy, freq, zwn)
    elif mode == 'numpy':
        if  _cmplx_so_exist_:
            pass
        elif root_method=='Fortran' and (not _cmplx_so_exist_):
            root_method='numpy'
            if iz == 0:
                print('$$$WARNING: cmplx_roots_sg.xxx.so not found, use numpy.roots instead$$$')
        return cal_ky_numpy(fu, fv, fqx, fqy, freq, zwn,
                            root_method=root_method)


