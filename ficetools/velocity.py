"""
A set of tools for the generation of velocity input data over a
specific domain.
"""

import logging
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from collections import defaultdict
import pandas as pd
import netCDF4
import os
import h5py
import pyproj
import salem

# Module logger
log = logging.getLogger(__name__)


def process_itslive_netcdf(dv, error_factor=1.0):
    """
    Open and process velocity netcdf for itslive
    it makes sure to mask ocean velocities equal to zero.
    :param dv: xarray dataset
    :param error_factor: error factor for which we
            enhance the std in the observations. default 1.0
    :return: vx, vy, std_vx, std_vx
    """

    # We do this in case we need to plot them at some point
    dv.attrs['pyproj_srs'] = dv.Polar_Stereographic.spatial_proj4
    for v in dv.variables:
        dv[v].attrs['pyproj_srs'] = dv.attrs['pyproj_srs']

    vx = dv.vx
    vy = dv.vy
    vx_err = dv.vx_err
    vy_err = dv.vy_err
    count = dv['count']

    nodata = -32767.0

    non_valid = (vx == nodata) | (vy == nodata)
    non_valid_e = (vx_err == nodata) | (vy_err == nodata)

    # We set invalid data (nans) to zero
    # After plotting the data we saw that nan's
    # were only found in the ocean where fenics_ice
    # does nothing...
    vx.data[non_valid] = 0.0
    vy.data[non_valid] = 0.0
    vx_err.data[non_valid_e] = 0.0
    vy_err.data[non_valid_e] = 0.0
    count.data[non_valid_e] = 0.0

    std_vx = (count ** (1 / 2)) * vx_err * error_factor
    std_vy = (count ** (1 / 2)) * vy_err * error_factor

    return vx, vy, std_vx, std_vy


def crop_velocity_data_to_extend(dvel, extend,
                                 return_coords=False,
                                 return_indexes=False,
                                 return_xarray=False):
    """
    Returns a xarray.Dataset crop to the given mesh
    extent
    :param
        dvel: xarray.Dataset to crop
        extend: extend to crop the data to
        given as a dictionary with the form:
        e.g. {'xmin': -1609000.0, 'xmax': -1381000.0,
              'ymin': -718450.0, 'ymax': -527000.0}
        return_coords: Bool to return x and y coordinates (default is False)
        return_indexes: Bool to return indexes of the cropped array (default is False)
        return_xarray: Bool to return the result as an xarray (default is False)
    :return
        see function below
    """

    # Processing vel data
    x_coords = dvel.x.data
    y_coords = dvel.y.data

    x_inds = np.where((x_coords >= extend["xmin"]) & (x_coords <= extend["xmax"]))[0]
    y_inds = np.where((y_coords >= extend["ymin"]) & (y_coords <= extend["ymax"]))[0]

    dv = dvel.isel(x=x_inds, y=y_inds)

    if return_xarray and return_indexes:
        return dv, x_inds, y_inds
    elif return_xarray:
        return dv
    elif return_coords:
        return dv.data, dv.x.data, dv.y.data
    else:
        return dv.data


def interpolate_missing_data(array, xx, yy):
    """
    Interpolate missing data via interpolate.gridata
    using nearest neighbour
    :param array: numpy array
    :param xx: 2d coord
    :param yy: 2d coord
    :return: flatten array
    """
    # mask invalid values
    array_ma = np.ma.masked_invalid(array)

    # get only the valid values
    x1 = xx[~array_ma.mask]
    y1 = yy[~array_ma.mask]
    newarr = array[~array_ma.mask]

    array_int = griddata((x1, y1), newarr.ravel(),
                   (xx, yy),
                   method='nearest')

    return array_int.ravel()


def open_and_crop_itslive_data(path_to_data, extend,
                               x_to_int, y_to_int):
    """
    Open and crop itslive data to the smith Glacier domain
    and puts them in the same grid as ASE

    :param path_to_data: string to itslive files
    :param extend: model domain extend
    :param x_to_int: x coordinate to interpolate to
    :param y_to_int: y coordinate to interpolate to
    :return: velocity components and uncertainty in the smith
    glacier domain and with the same resolution than ASE

    """
    dvel = xr.open_dataset(path_to_data)

    vel_obs = dvel.v
    vx = dvel.vx
    vy = dvel.vy
    vx_err = dvel.vx_err
    vy_err = dvel.vy_err

    # Crop to smith Glacier extend
    vx_s, xind, yind = crop_velocity_data_to_extend(vx, extend, return_xarray=True, return_indexes=True)
    vy_s = crop_velocity_data_to_extend(vy, extend, return_xarray=True)
    errx_s = crop_velocity_data_to_extend(vx_err, extend, return_xarray=True)
    erry_s = crop_velocity_data_to_extend(vy_err, extend, return_xarray=True)

    vel_obs_s = crop_velocity_data_to_extend(vel_obs, extend, return_xarray=True)

    # Interpolate ITSlive 2011 data to the same resolution of ASE
    vx_int = vx_s.interp(x=x_to_int, y=y_to_int)
    vy_int = vy_s.interp(x=x_to_int, y=y_to_int)
    errx_int = errx_s.interp(x=x_to_int, y=y_to_int)
    erry_int = erry_s.interp(x=x_to_int, y=y_to_int)

    vel_obs_int = vel_obs_s.interp(x=x_to_int, y=y_to_int)

    return vel_obs_int, vx_int, vy_int, errx_int, erry_int


def check_if_arrays_have_same_shape(arrays, array_shape):
    """
    Returns a list of bools checking if all arrays have the same
    shape
    :param arrays: list of arrays to check
    :param array_shape: shape to check against
    :return: a list of True or False
    """
    bools = []
    for array in arrays:
        if array.shape == array_shape:
            bools.append(True)
        else:
            bools.append(False)
    return bools


def drop_invalid_data_from_several_arrays(x, y,
                                          vx, vy,
                                          vx_err, vy_err, masked_array):
    """
    Drops nan data from multiple xarray.datasets containing velocity
    information

    :param x: 1d array of coordinates to for a grid
    :param y: 1d array of coordinates to for a grid
    :param vx: velocity component in x direction
    :param vy: velocity component in y direction
    :param vx_err: velocity component error in x direction
    :param vy_err: velocity component error in y direction
    :param masked_array: numpy.mask array
    :return: variables without nans
    """

    # Now we drop invalid data
    x_grid, y_grid = np.meshgrid(x, y)

    x_nonan = x_grid[~masked_array.mask]
    y_nonan = y_grid[~masked_array.mask]

    vy_nonan = vy.data[~masked_array.mask]
    vx_nonan = vx.data[~masked_array.mask]

    vy_err_nonan = vy_err.data[~masked_array.mask]
    vx_err_nonan = vx_err.data[~masked_array.mask]

    shape_after = vy_nonan.shape
    print('Shape after nan drop mus be values,')
    print(shape_after)

    all_data = [x_nonan, y_nonan,
                vy_nonan, vx_nonan,
                vy_err_nonan, vx_err_nonan]

    bool_list = check_if_arrays_have_same_shape(all_data,
                                                shape_after)

    assert all(element == True for element in bool_list)
    return x_nonan, y_nonan, vx_nonan, vy_nonan, vx_err_nonan, vy_err_nonan


def compute_coarsen(data, resolution):
    """
    Create a new xarray data set from Itslive
     with a lower resolution

    :param data: original DataFrame from Itslive
    :param resolution: desired resolution as int
    :return: new dataframe with a lower resolution
    """

    data_res = float(abs(data.x[0] - data.x[1]))

    # Just take one var data
    vx = data.vx
    vx_attrs = data.vx.attrs
    vy = data.vy
    vy_attrs = data.vy.attrs

    vx_err = data.vx_err
    vx_err_attrs = data.vx_err.attrs
    vy_err = data.vy_err
    vy_err_attrs = data.vy_err.attrs

    count = data['count']
    count_attrs = data['count'].attrs

    # Average the data array in a coarse resolution
    # A window for avg. needs to be supplied e.g 2.0/res = 40
    vx_c = vx.coarsen({'y': int(resolution / data_res),
                       'x': int(resolution / data_res)}, boundary='pad').mean()

    vy_c = vy.coarsen({'y': int(resolution / data_res),
                       'x': int(resolution / data_res)}, boundary='pad').mean()

    vx_err_c = vx_err.coarsen({'y': int(resolution / data_res),
                       'x': int(resolution / data_res)}, boundary='pad').mean()

    vy_err_c = vy_err.coarsen({'y': int(resolution / data_res),
                               'x': int(resolution / data_res)}, boundary='pad').mean()

    count_c = count.coarsen({'y': int(resolution / data_res),
                               'x': int(resolution / data_res)}, boundary='pad').mean()

    #  Creating a new data set
    ds = xr.Dataset({'vx': (['y', 'x'], vx_c.data),
                     'vy': (['y', 'x'], vy_c.data),
                     'vx_err': (['y', 'x'], vx_err_c.data),
                     'vy_err': (['y', 'x'], vy_err_c.data),
                     'count': (['y', 'x'], count_c.data)
                     },
                    coords={'y': (['y'], vx_c.y.values),
                            'x': (['x'], vx_c.x.values)}
                    )

    ds['vx'].attrs = vx_attrs
    ds['vy'].attrs = vy_attrs
    ds['vx_err'].attrs = vx_err_attrs
    ds['vy_err'].attrs = vy_err_attrs
    ds['count'].attrs = count_attrs

    dic = data.attrs
    ds.attrs = dic

    return ds


def interp_to_measures_grid(dv, dm):
    """
    Interpolate velocity data to measures grid, for velocity
    comparisons

    :param dv: the other velocity data set (e.g. itslive)
    :param dm: measures velocity data set
    :return: xarray.Dataset containing the new interpolated data
    to measures resolution
    """

    new_x = dm.x.values
    new_y = dm.y.values

    array_ma = np.ma.masked_invalid(dm.VX.values)

    res = abs(dm.x[0] - dm.x[1])

    # Just take one var data
    vx = dv.vx
    vx_attrs = dv.vx.attrs
    vy = dv.vy
    vy_attrs = dv.vy.attrs

    vx_err = dv.vx_err
    vx_err_attrs = dv.vx_err.attrs
    vy_err = dv.vy_err
    vy_err_attrs = dv.vy_err.attrs

    count = dv['count']
    count_attrs = dv['count'].attrs

    vx_new = vx.interp(x=new_x, y=new_y)
    vy_new = vy.interp(x=new_x, y=new_y)
    vx_err_new = vx_err.interp(x=new_x, y=new_y)
    vy_err_new = vy_err.interp(x=new_x, y=new_y)
    count_new = count.interp(x=new_x, y=new_y)

    vx_new.data[array_ma.mask] = np.NaN
    vy_new.data[array_ma.mask] = np.NaN
    vx_err_new.data[array_ma.mask] = np.NaN
    vy_err_new.data[array_ma.mask] = np.NaN
    count_new.data[array_ma.mask] = np.NaN

    #  Creating a new data set
    ds = xr.Dataset({'vx': (['y', 'x'], vx_new.data),
                     'vy': (['y', 'x'], vy_new.data),
                     'vx_err': (['y', 'x'], vx_err_new.data),
                     'vy_err': (['y', 'x'], vy_err_new.data),
                     'count': (['y', 'x'], count_new.data)
                     },
                    coords={'y': (['y'], vx_new.y.values),
                            'x': (['x'], vx_new.x.values)}
                    )
    return ds


def create_subsample(ds, step):
    """
    Creates a subsample of the velocity data by hovering over the original array
    and cropping the data every certain step, we only retain the top left element
    of that subarray.

    :param ds: xarray.Dataset containing a velocity or velocity error vector
            (eg. vx, vy, stdvx, stdvy)
    :param step: step for subsampling (eg. every 10 grid spaces)
    :param return_coords: if true returns coordinates for the new arrays
    :return: b_0 and b_m: the top left or middle element of the subarray
    """

    x = ds.x.data
    y = ds.y.data
    sel = ds.data

    x_0 = x[::step]
    y_0 = y[::step]
    x_m = x[(step // 2)::step]
    y_m = y[(step // 2)::step]

    b_m = sel[(step // 2)::step, (step // 2)::step]
    b_0 = sel[::step, ::step]

    [xx_0, yy_0] = np.meshgrid(x_0, y_0)
    [xx_m, yy_m] = np.meshgrid(x_m, y_m)

    coords_0 = tuple(zip(xx_0, yy_0))
    coords_m = tuple(zip(xx_m, yy_m))

    assert not np.array_equal(coords_0, coords_m)

    return (xx_0.flatten(), yy_0.flatten(), b_0.flatten()), \
           (xx_m.flatten(), yy_m.flatten(), b_m.flatten())


def create_adjusted_std_maxima(std_o, std_to_a):
    """
    Makes a new std estimate of the velocity component (std_vx or
    std_vy), by selecting the maximum value between the original estimate
    (std_o) and the adjusted estimate given to the function (std_to_a)

    :param std_o: xarray.DataArray with original std values
    :param std_to_a: xarray.DAtaArray containing the value that will
        adjust the std to in case that such value is higher than
        the original (std_o).
    :return: ds: xarray.DataArray with adjusted std and the nans
    filled with values from the original std array (std_o)
    """

    ds = np.maximum(std_o, std_to_a)

    # array to mask ... a dot product of component and std
    mask_array = ds.data

    # Remove nan from cloud data
    array_ma = np.ma.masked_invalid(mask_array)

    x_s = std_o.x.data
    y_s = std_o.y.data

    x_grid, y_grid = np.meshgrid(x_s, y_s)

    x_nan = x_grid[array_ma.mask].ravel()
    y_nan = y_grid[array_ma.mask].ravel()

    for x, y in zip(x_nan, y_nan):
        ds.loc[dict(x=x, y=y)] = std_o.loc[dict(x=x, y=y)]

    return ds


def drop_nan_from_multiple_numpy(x, y,
                                 vx, vy,
                                 std_vx, std_vy):
    """
    Drops the np.nan points generated by the subsampling and returns tuples
    for x, y and velocity data components.

    :param x: x-coordinates
    :param y: y-coordinates
    :param vx: velocity x component
    :param vy: velocity y component
    :param std_vx: std of the x velocity component
    :param std_vy: std of the y velocity component
    :return: pandas.Dataframe with all the velocity components with no nan
    """
    arrays = {'y': y, 'x': x,
              'vx': vx, 'vy': vy,
              'std_vx': std_vx, 'std_vy': std_vy}

    df = pd.DataFrame(data=arrays)
    df_new = df.dropna(subset=['vx', 'vy'], how='all')
    df_new = df_new.dropna(subset=['std_vx', 'std_vy'], how='all')

    return df_new


def drop_nan_from_xarray(ds, return_coords=False):
    """
    Drops the np.nan points generated by the subsampling and returns tuples
    for x, y and velocity data components.
    :param ds:
    :param return_coords:
    :return: numpy.ndarray with the velocity information without nan
    if return_coords is true also returns the coordinates
    """
    x = ds.x.values
    y = ds.y.values

    masked_array = np.ma.masked_invalid(ds.data)

    x_grid, y_grid = np.meshgrid(x, y)

    x_nonan = x_grid[~masked_array.mask]
    y_nonan = y_grid[~masked_array.mask]

    ds_nonan = masked_array.data[~masked_array.mask]

    shape_after = ds_nonan.shape

    all_data = [x_nonan, y_nonan,
                ds_nonan]

    bool_list = check_if_arrays_have_same_shape(all_data,
                                                shape_after)

    assert all(element == True for element in bool_list)

    if return_coords:
        return x_nonan, y_nonan, ds_nonan
    return ds_nonan

def define_salem_grid_from_measures(fp_vel, model_extend):
    """
    Define a Salem grid according to MEaSUREs mosaic
    velocity file.
    It crops the data to the model domain and defines a
    salem grid to plot the data.

    :param fp_vel: path to MEaSUREs file
    :param model_extend: extend of the model where to crop
    the data
    :return: salem.Grid as g
    """

    dv = xr.open_dataset(fp_vel)

    vx = dv.VX
    vy = dv.VY
    std_vx = dv.STDX
    std_vy = dv.STDY

    # Crop velocity data to the Smith Glacier extend
    vx_s = crop_velocity_data_to_extend(vx, model_extend, return_xarray=True)

    # Lets define our salem grid. (but we modified things cuz
    # fabi's code only works for the North! TODO: ask fabien)

    proj = pyproj.Proj('EPSG:3413')
    y_grid = vx_s.y
    x_grid = vx_s.x

    dy = abs(y_grid[0] - y_grid[1])
    dx = abs(x_grid[0] - x_grid[1])

    # Pixel corner
    origin_y = y_grid[0] + dy * 0.5
    origin_x = x_grid[0] - dx * 0.5

    gv = salem.Grid(nxny=(len(x_grid), len(y_grid)), dxdy=(dx, -1 * dy),
                    # We use -dy as this is the Sout Hemisphere somehow salem
                    x0y0=(origin_x, origin_y), proj=proj)  # is not picking that up!

    return gv

def write_velocity_tuple_h5file(comp_dict, cloud_dict, fpath=str):
    """
    Write velocity tuple h5 file with composite data and cloud data
    Before writing any data a lot of checks are done.

    :param comp_dict: dictionary containing composite velocity arrays
    :param cloud_dict: dictionary containing cloud velocity arrays
    :param fpath: path to save the data to.
    :return: writes a new velocity file in the required directory
    """

    # Read the composite dictionary
    vx_comp = comp_dict['vx_comp']
    errx_comp = comp_dict['std_vx_comp']
    vy_comp = comp_dict['vy_comp']
    erry_comp = comp_dict['std_vy_comp']

    x_comp = comp_dict['x_comp']
    y_comp = comp_dict['y_comp']

    mask_comp = np.array(vx_comp, dtype=bool)

    # Sanity checks for composite data!
    assert np.count_nonzero(np.isnan(vx_comp)) == 0
    assert np.count_nonzero(np.isnan(vy_comp)) == 0
    assert np.count_nonzero(np.isnan(errx_comp)) == 0
    assert np.count_nonzero(np.isnan(erry_comp)) == 0

    all_data = [x_comp, y_comp,
                vx_comp, vy_comp,
                errx_comp, erry_comp]
    shape_after = mask_comp.shape
    bool_list = check_if_arrays_have_same_shape(all_data, shape_after)
    assert all(element == True for element in bool_list)

    # Read the cloud dictionary

    vx_cloud = cloud_dict['vx_cloud']
    vx_err_cloud = cloud_dict['std_vx_cloud']
    vy_cloud = cloud_dict['vy_cloud']
    vy_err_cloud = cloud_dict['std_vy_cloud']

    x_cloud = cloud_dict['x_cloud']
    y_cloud = cloud_dict['y_cloud']

    # Sanity checks!
    if vx_cloud is not None:
        assert np.count_nonzero(np.isnan(vx_cloud)) == 0
        assert np.count_nonzero(np.isnan(vy_cloud)) == 0
        assert np.count_nonzero(np.isnan(vx_err_cloud)) == 0
        assert np.count_nonzero(np.isnan(vy_err_cloud)) == 0
        all_data = [x_cloud, y_cloud,
                    vx_cloud, vy_cloud,
                    vx_err_cloud, vy_err_cloud]
        shape_after = x_cloud.shape
        bool_list = check_if_arrays_have_same_shape(all_data, shape_after)
        assert all(element == True for element in bool_list)

    if os.path.exists(fpath):
        os.remove(fpath)
    else:
        print("The file did not exist before so is being created now")

    with h5py.File(fpath, 'w') as outty:
        data = outty.create_dataset("mask_vel", mask_comp.shape, dtype='f')
        data[:] = mask_comp
        data = outty.create_dataset("u_obs", vx_comp.shape, dtype='f')
        data[:] = vx_comp
        data = outty.create_dataset("u_std", errx_comp.shape, dtype='f')
        data[:] = errx_comp
        data = outty.create_dataset("v_obs", vy_comp.shape, dtype='f')
        data[:] = vy_comp
        data = outty.create_dataset("v_std", erry_comp.shape, dtype='f')
        data[:] = erry_comp
        data = outty.create_dataset("x", x_comp.shape, dtype='f')
        data[:] = x_comp
        data = outty.create_dataset("y", y_comp.shape, dtype='f')
        data[:] = y_comp
        if vx_cloud is not None:
            data = outty.create_dataset("u_cloud", vx_cloud.shape, dtype='f')
            data[:] = vx_cloud
            data = outty.create_dataset("u_cloud_std", vx_err_cloud.shape, dtype='f')
            data[:] = vx_err_cloud
            data = outty.create_dataset("v_cloud", vy_cloud.shape, dtype='f')
            data[:] = vy_cloud
            data = outty.create_dataset("v_cloud_std", vy_err_cloud.shape, dtype='f')
            data[:] = vy_err_cloud
            data = outty.create_dataset("x_cloud", x_cloud.shape, dtype='f')
            data[:] = x_cloud
            data = outty.create_dataset("y_cloud", y_cloud.shape, dtype='f')
            data[:] = y_cloud


def merge_measures_and_itslive_vel_obs_sens(dic_il, dic_me, return_df_merge=False):
    """
    Function to merge each mesasures vel obs sens data
    to the same amount of velocity points as the itslive data set
    dig_il: dictionary with vel obs sens data for a simulation with itslive
    dig_me: dictionary with vel obs sens data for a simulation with measures
    return_df_merge: default is False if you dont want the Pandas dataframe merged
    """
    xm, ym = np.split(dic_me['uv_obs_pts'], 2, axis=1)
    xi, yi = np.split(dic_il['uv_obs_pts'], 2, axis=1)

    d_il = {'xi': xi.ravel(),
            'yi': yi.ravel(),
            'u_obs_i': dic_il['u_obs'],
            'v_obs_i': dic_il['v_obs']}

    df_il = pd.DataFrame(data=d_il)

    d_me = {'xm': xm.ravel(),
            'ym': ym.ravel(),
            'u_obs_m': dic_me['u_obs'],
            'v_obs_m': dic_me['v_obs']}

    df_me = pd.DataFrame(data=d_me)

    df_merge = pd.merge(df_il,
                        df_me,
                        how='inner',
                        left_on=['xi', 'yi'],
                        right_on=['xm', 'ym'])

    # we will save all dataframes on a dict
    all_dfs = defaultdict(list)

    for n_sen in np.arange(15):
        dict_me = {'xm': xm.ravel(),
                   'ym': ym.ravel(),
                   'dObsU_me': dic_me['dObsU'][n_sen],
                   'dObsV_me': dic_me['dObsV'][n_sen]}

        df_measures = pd.DataFrame(data=dict_me)

        dict_il = {'xi': xi.ravel(),
                   'yi': yi.ravel(),
                   'dObsU_il': dic_il['dObsU'][n_sen],
                   'dObsV_il': dic_il['dObsV'][n_sen]}

        df_itslive = pd.DataFrame(data=dict_il)

        df_merge_dqdo = pd.merge(df_itslive,
                                 df_measures,
                                 how='inner',
                                 left_on=['xi', 'yi'],
                                 right_on=['xm', 'ym'])

        # print(df_merge_dqdo.keys())
        assert df_merge_dqdo.shape == df_merge.shape

        all_dfs[n_sen] = df_merge_dqdo

    if return_df_merge:
        return all_dfs, df_merge
    else:
        return all_dfs


def dot_product_per_pair(all_dfs, df_obs):
    """
    Compute derivatives dot product, per pair of data sets (itslive or measures)
    :param all_dfs: dataframe containing itslive and measures vel obs sens data
    :df_obs: dataframe with velocity observations
    """
    dot_product_U_me = []
    dot_product_V_me = []

    for n_sen in np.arange(15):
        dq_du_me = all_dfs[n_sen]['dObsU_me']
        dq_dv_me = all_dfs[n_sen]['dObsV_me']

        u_il = df_obs['u_obs_il']
        v_il = df_obs['v_obs_il']

        u_me = df_obs['u_obs_me']
        v_me = df_obs['v_obs_me']

        dq_du_dot_me = np.dot(dq_du_me, u_me - u_il)
        dq_dv_dot_me = np.dot(dq_dv_me, v_me - v_il)

        dot_product_U_me.append(dq_du_dot_me)
        dot_product_V_me.append(dq_dv_dot_me)

    return dot_product_U_me, dot_product_V_me

class ncDataset(netCDF4.Dataset):
    """Wrapper around netCDF4 setting auto_mask to False"""

    def __init__(self, *args, **kwargs):
        super(ncDataset, self).__init__(*args, **kwargs)
        self.set_auto_mask(False)