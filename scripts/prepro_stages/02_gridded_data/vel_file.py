"""
Crop MEaSUREs velocity files to the Smith glacier extend
and save the data in a .h5 format.

With the following variables stored as a could point data containing
no np.nan's:
- 'u_obs', 'u_std', 'v_obs', 'v_std', 'x', 'y'.

each variable with shape: (#values, )

We also store a grid extend as dictionary to perform
nearest neighbor interpolation of missing velocities
from within Fenics_ice

@authors: Fenics_ice contributors
"""
import xarray as xr
import os
import sys
from configobj import ConfigObj
import numpy as np
import h5py

# Set path to working directory and meshtools
MAIN_PATH = os.path.expanduser('~/scratch/smith_glacier/')
sys.path.append(MAIN_PATH)
from meshtools import meshtools as mesht

# Load configuration file for more order in paths
config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Load the data
dvel = xr.open_dataset(os.path.join(MAIN_PATH,
                                    config['velocity_netcdf']))

# We take everything in 2010
year = '2010'
list_vars = ['vx', 'vy', 'err']

keys = []
for var in list_vars:
    keys.append(f'{var}{year}')

vx = dvel[keys[0]]
vy = dvel[keys[1]]
err = dvel[keys[2]]

x = dvel['xaxis']
y = dvel['yaxis']

# Calculating vector magnitude
vel_obs = np.sqrt(vx**2 + vy**2).rename('vel_obs')

# Now we need the error components of the error magnitude vector
# To find them we first we need to compute the vector direction
# I am assuming is the same as the velocity vector
vel_err_dir = np.arctan(vy/vx) #* 180/np.pi not sure about the angle

# err-y component
err_y = err*np.cos(vel_err_dir).rename('err_y')
# err-x component
err_x = err*np.sin(vel_err_dir).rename('err_y')

# We re-compute the error magnitude to check that the
# error array is the same...
# Surprise! it wont be the same as vy and vx have more nans than
# the err vector.. these nans are passed to vel_err_dir...
# we end up with lest valid data points in err due to that.
# But is weird that nan's in the velocity data
# can have error estimates?
err_obs_check = np.sqrt(err_x**2 + err_y**2)
err_obs_check = err_obs_check.rename('err2010')

smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -717500.0,
              'ymax': -528500.0}

x_inds = np.where((x >= smith_bbox["xmin"]) & (x <= smith_bbox["xmax"]))[0]
y_inds = np.where((y >= smith_bbox["ymin"]) & (y <= smith_bbox["ymax"]))[0]

x_slice = x[x_inds]
y_slice = y[y_inds]

grid_extend = {'xmin':np.min(x_slice.data), 'ymin': np.min(y_slice.data),
               'xmax':np.max(x_slice.data), 'ymax': np.max(y_slice.data),
               'dx': abs(x_slice.data[1]-x_slice.data[0]),
               'dy': abs(y_slice.data[1]-y_slice.data[0])}

vx_slice = vx[y_inds, x_inds]
vy_slice = vy[y_inds, x_inds]
err_slice = err_obs_check[y_inds, x_inds]
err_y_slice = err_y[y_inds, x_inds].rename('err_y')
err_x_slice = err_x[y_inds, x_inds].rename('err_x')
vel_obs_slice = vel_obs[y_inds, x_inds]

# We get rid of outliers in the data
y_out, x_out = np.where(vel_obs_slice.data > 5000)
vel_obs_slice[y_out, x_out] = np.nan
vx_slice[y_out, x_out] = np.nan
vy_slice[y_out, x_out] = np.nan
err_slice[y_out, x_out] = np.nan
err_y_slice[y_out, x_out] = np.nan
err_x_slice[y_out, x_out] = np.nan

shape_before = vy_slice.shape
print('Shape before nan drop')
print(shape_before)

# Now we drop invalid data
x_grid, y_grid = np.meshgrid(x_slice, y_slice)

vel_ma = np.ma.masked_invalid(vel_obs_slice.data)
print('to check invalid shape')
print(vel_ma.shape)

x_nonan = x_grid[~vel_ma.mask]
y_nonan = y_grid[~vel_ma.mask]

vel_nonan = vel_obs_slice.data[~vel_ma.mask]
vy_nonan = vy_slice.data[~vel_ma.mask]
vx_nonan = vx_slice.data[~vel_ma.mask]

err_nonan = err_slice.data[~vel_ma.mask]
err_y_nonan = err_y_slice.data[~vel_ma.mask]
err_x_nonan = err_x_slice.data[~vel_ma.mask]

shape_after = vy_nonan.shape
print('Shape after')
print(shape_after)

all_data = [x_nonan, y_nonan,
            vel_nonan, vy_nonan, vx_nonan,
            err_nonan, err_y_nonan, err_x_nonan]

bool_list = mesht.check_if_arrays_have_same_shape(all_data,
                                                  shape_after)

assert all(element==True for element in bool_list)

mask = np.array(vel_nonan, dtype=bool)

file_name = os.path.join(MAIN_PATH,
                         config['smith_vel_obs'])

with h5py.File(file_name, 'w') as outty:
    data = outty.create_dataset("mask_vel", mask.shape, dtype='f')
    data[:] = mask
    data = outty.create_dataset("u_obs", vx_nonan.shape, dtype='f')
    data[:] = vx_nonan
    data = outty.create_dataset("u_std", err_x_nonan.shape, dtype='f')
    data[:] = err_x_nonan
    data = outty.create_dataset("v_obs", vy_nonan.shape, dtype='f')
    data[:] = vy_nonan
    data = outty.create_dataset("v_std", err_y_nonan.shape, dtype='f')
    data[:] = err_y_nonan
    data = outty.create_dataset("x", x_nonan.shape, dtype='f')
    data[:] = x_nonan
    data = outty.create_dataset("y", y_nonan.shape, dtype='f')
    data[:] = y_nonan
    for k in grid_extend.keys():
        outty.attrs[k] = grid_extend[k]

