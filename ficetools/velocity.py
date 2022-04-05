"""
A set of tools for the generation of velocity input data over a
specific domain.
"""

import logging
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

# Module logger
log = logging.getLogger(__name__)

def crop_velocity_data_to_extend(dvel, extend,
                                 return_coords=False,
                                 return_indexes=False,
                                 return_xarray=False):
    """
    Returns a xarray.Dataset crop to the given
    extent
    :param
        dvel: xarray.Dataset to crop
        extend: extend to crop the data to
        given as a dictionary with the form:
        e.g. {'xmin': -1609000.0, 'xmax': -1381000.0,
              'ymin': -718450.0, 'ymax': -527000.0}
        return_coords: Bool to return x and y coordinates
        default is False
    :return
        dv.data: numpy array with data
        dv.x.data: x coordinates
        dv.y.data: y coordinates
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
    Opens and crops itslive data to the smith Glacier domain
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

    vel_nonan = masked_array.data[~masked_array.mask]
    vy_nonan = vy[~masked_array.mask]
    vx_nonan = vx[~masked_array.mask]

    vy_err_nonan = vy_err[~masked_array.mask]
    vx_err_nonan = vx_err[~masked_array.mask]

    shape_after = vy_nonan.shape
    print('Shape after nan drop mus be values,')
    print(shape_after)

    all_data = [x_nonan, y_nonan,
                vel_nonan, vy_nonan, vx_nonan,
                vy_err_nonan, vx_err_nonan]

    bool_list = check_if_arrays_have_same_shape(all_data,
                                                shape_after)

    assert all(element == True for element in bool_list)
    return x_nonan, y_nonan, vx_nonan, vy_nonan, vx_err_nonan, vy_err_nonan, vel_nonan
