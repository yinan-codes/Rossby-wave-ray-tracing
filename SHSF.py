# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:27:28 2025

@author: 杨艺楠
"""

import pyshtools as pysh

def SHSF(data, truncation_level, sampling=2):
    """
    球面谐函数谱滤波（Spherical Harmonic Spectral Filtering）方法
    对输入数据进行球谐展开、截断和重建
    
    参数:
    - data (ndarray): 输入的网格数据 (2D 或 3D)
    - truncation_level (int): 截断的球谐级数
    - sampling (int): 采样方式，默认使用 Driscoll and Healy (DH), 即球面等距采样 (2)
    
    返回:
    - reconstructed (ndarray): 重建后的网格数据
    """
    # Step 1: 进行球谐展开 (相当于 shaec)
    coeffs = pysh.expand.SHExpandDH(data, sampling=sampling)
    
    # Step 2: 创建 SHCoeffs 对象并进行截断 (相当于 tri_trunc)
    coeffs_obj = pysh.SHCoeffs.from_array(coeffs)
    truncated_coeffs = coeffs_obj.pad(truncation_level)
    
    # Step 3: 通过截断后的球谐系数重建网格 (相当于 shsec)
    reconstructed = pysh.expand.MakeGridDH(truncated_coeffs.to_array())
    
    return reconstructed


'''
# 示例用法
truncation_level = 44  # 根据需要设定截断级数 Lmax≈180°/Δϕ−1
u_reconstructed = SHSF(ubar, truncation_level)
v_reconstructed = SHSF(vbar, truncation_level)
'''
