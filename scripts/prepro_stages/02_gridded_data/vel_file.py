"""
Crop MEaSUREs velocity files to the Smith glacier extend
and save the data with a .h5.

With the following variables as a list of values with no np.nan's:
'mask_vel', 'u_obs', 'u_std', 'v_obs', 'v_std', 'x', 'y'
and with variable.shape: (#values, )

@authors: Fenics_ice contributors
"""
import xarray as xr
import os
import sys
from configobj import ConfigObj
import numpy as np
import h5py

# Set path to working directory and meshtools
MAIN_PATH = os.path.expanduser('~/smith_glacier/')
sys.path.append(MAIN_PATH)
from meshtools import meshtools as meshtools

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

# Now we need the error components of the
# error magnitude vector
# But first we need the vector direction
# I am asuming is the same as the velocity
# vector
vel_err_dir = np.arctan(vy/vx) #* 180/np.pi not sure about the angle

# err-y component
err_y = err*np.cos(vel_err_dir).rename('err_y')
# err-x component
err_x = err*np.sin(vel_err_dir).rename('err_y')

# Computing the magnitude as if we use err we will end up
# with more pixels with data than if we use the components
# See notebook for more info on this
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

vx_slice = vx[y_inds, x_inds]
vy_slice = vy[y_inds, x_inds]
err_slice = err_obs_check[y_inds, x_inds]
err_y_slice = err_y[y_inds, x_inds].rename('err_y')
err_x_slice = err_x[y_inds, x_inds].rename('err_x')
vel_obs_slice = vel_obs[y_inds, x_inds]

shape_before = vx_slice.shape
print('Shape before nan drop')
print(shape_before)

vx_f, nx, ny = meshtools.dropnan_values(vx_slice, get_xy=True)

vy_f = meshtools.dropnan_values(vy_slice)
v_f = meshtools.dropnan_values(vel_obs_slice)

err_x_f = meshtools.dropnan_values(err_x_slice)
err_y_f = meshtools.dropnan_values(err_y_slice)
err_f = meshtools.dropnan_values(err_slice)

x_nonan = x_slice[nx]
y_nonan = y_slice[ny]

shape_after = vx_f.shape
print(shape_after)
all_data = [v_f, vx_f, vy_f, err_x_f, err_y_f, err_f, x_nonan, y_nonan]
bool_list = meshtools.check_if_arrays_have_same_shape(all_data,
                                                  shape_after)

assert all(element==True for element in bool_list)

mask = np.array(v_f, dtype=bool)

file_name = os.path.join(MAIN_PATH,
                         config['smith_vel_obs'])

with h5py.File(file_name, 'w') as outty:
    data = outty.create_dataset("mask_vel", mask.shape, dtype='f')
    data[:] = mask
    data = outty.create_dataset("u_obs", vx_f.shape, dtype='f')
    data[:] = vx_f
    data = outty.create_dataset("u_std", err_x_f.shape, dtype='f')
    data[:] = err_x_f
    data = outty.create_dataset("v_obs", vy_f.shape, dtype='f')
    data[:] = vy_f
    data = outty.create_dataset("v_std", err_y_f.shape, dtype='f')
    data[:] = err_y_f
    data = outty.create_dataset("x", x_nonan.shape, dtype='f')
    data[:] = x_nonan
    data = outty.create_dataset("y", y_nonan.shape, dtype='f')
    data[:] = y_nonan
