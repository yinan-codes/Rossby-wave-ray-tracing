import numpy as np
import warnings

warnings.filterwarnings("ignore")
parameters = {
    'freq': 0., # frequency; Set to 0 for stationary Rossby waves.
    'mm': None, # mm: nlon; None: read nc to get
    'nn': None, # nn: nlat; None: read nc to get
    'SW_lon': 70., 'SW_lat': -4., # lon and lat at south-west corner of source matrix
    'dlon': 4, 'dlat': 2,  # lon and lat step of source matrix
    'nnx': 21, 'nny': 15, # source point nums on lon and lat    # For zonal mean basic flow, the number of longitudes must be 
                                                                # greater than or equal to 2 for sake of derivatives
    'zwn': np.array([1., 2., 3., 4., 5., 6., 7.]), # initial zonal wave number
    'nzwn': 7, # zonal wave number, decided by len(zwn)
    'tstep': 2, # time step, unit: hour
    'ttotal': 90., # unit: day
    'mode': 'numpy', 
    'root_method': 'numpy', 
    'inte_method': '', # integral method, ''(defualt rk4) or 'rk45'
    'xcyclic': True,
    'cal_dtype':'float64', # now only support float64
    'read_dtype': 'float32', # dtype read from nc
    'inputuv' : 'D:/data/to/your/path.nc', # the Basic flow
    'bsfile' : 'D:/path/to/be/stored.nc', # the full suite of basic flow diagnostics
    'ncfile'  : 'D:/path/to/be/stored.nc', # ray tracing result file

    'rtol': 1e-6,
    'atol': 1e-6, # Relative and absolute tolerances, for rk45
    'MinStepFactor':1e-3, # contorl minstep for rk45 = tstep*msf
}
def real2d_hnf(
    nzwn,
    mm,
    nn,
    freq,
    zwn,
    inputuv,
    ncfile,
    bsfile,
    SW_lon,
    SW_lat,
    dlon,
    dlat,
    nnx,
    nny,
    mode,
    tstep,
    ttotal,
    xcyclic,
    root_method, # path_to_os,
    read_dtype,
    cal_dtype,
    inte_method,
    atol,
    rtol,
    MinStepFactor
):
    # if root_method == 'Fortran':
    #     add_so_path(path_to_os)

    from constants import hour, day
    from wr import WR

    # freq = - 1.D0 / ( 5.D0 * day)
    nsource = nnx * nny
    wr1 = WR(
        nzwn,
        nsource,
        tstep * hour,
        ttotal * day,
        freq,
        nx=mm,
        ny=nn,
        read_dtype=read_dtype,
        cal_dtype=cal_dtype,
        rtol=rtol,
        atol=atol,
        ncfile=inputuv,
        MinStepFactor=MinStepFactor)
    wr1.bs.loadbs_ncfile(inputuv)
    wr1.bs.ready(xcyclic=xcyclic)
    wr1.bs.output(bsfile)
    # wr1.bs.read_from_precaled_nc(r'/home/bsfortran.nc')
    wr1.set_zwn(zwn)
    wr1.set_source_matrix(SW_lon, SW_lat, dlon, dlat, nnx, nny)
    wr1.ray_info()
    # ,debug=True,debug_file=r'/home/output.nc')
    wr1.ray_run(mode=mode, root_method=root_method, inte_method=inte_method)
    wr1.output(ncfile)

    # # deallocate the arrays
    # call wr1%clean
def add_so_path(path):
    import sys
    if path not in sys.path:
        sys.path.append(path)


if __name__ == '__main__':
    real2d_hnf(**parameters)
