"""
Crop ASE time series and ITSlive composite mosaic
of ice velocities to the Smith glacier extend
and save the data in a .h5 format.

With the following variables stored as a could point data containing
no np.nan's:
- 'u_obs', 'u_std', 'v_obs', 'v_std', 'x', 'y'.
- 'u_comp', 'u_comp_std', 'v_comp', 'v_comp_std', 'x_comp', 'y_comp', 'mask_vel_comp'

each variable with shape: (#values, )

@authors: Fenics_ice contributors
"""
import xarray as xr
import salem
import os
import sys
from configobj import ConfigObj
import numpy as np
import h5py

# Define main repository path
MAIN_PATH = os.path.expanduser('~/scratch/smith_glacier/')
sys.path.append(MAIN_PATH)
# Load configuration file for more order in paths
config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

from meshtools import meshtools as meshtools

# Define the Smith Glacier extent to crop all
# velocity data to this region
# IMPORTANT .. verify that the extent is always
# bigger than the mesh!
smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -718450.0,
              'ymax': -527000.0}

# 1) First load and process ITSLive data for storing
# a composite mean of all velocity components and uncertainty
path_itslive = os.path.join(MAIN_PATH, config['itslive'])
file_names = os.listdir(path_itslive)

paths_itslive = []
for f in file_names:
    paths_itslive.append(os.path.join(path_itslive, f))

print(paths_itslive)

# Opening files with salem slower than rasterio
# but they end up more organised in xr.DataArrays
dvx = salem.open_xr_dataset(paths_itslive[0])
dvx_err = salem.open_xr_dataset(paths_itslive[1])
dvy = salem.open_xr_dataset(paths_itslive[2])
dvy_err = salem.open_xr_dataset(paths_itslive[3])

vx = dvx.data
vy = dvy.data
vx_err = dvx_err.data
vy_err = dvy_err.data

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
vx_err.data[non_valid_e] = 0.0

vx_s = meshtools.crop_itslive_raster(dvx, smith_bbox)
vy_s = meshtools.crop_itslive_raster(dvy, smith_bbox)
vx_err_s = meshtools.crop_itslive_raster(dvx_err, smith_bbox)
vy_err_s = meshtools.crop_itslive_raster(dvy_err, smith_bbox)

y_il = vx_s.y.values
x_il = vx_s.x.values

print('Lets check all has the same shape')
print(vx_s.shape, vy_s.shape)
print(vx_err_s.shape, vy_err_s.shape)
print(y_il.shape)
print(x_il.shape)

x_grid_il, y_grid_il = np.meshgrid(x_il, y_il)

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
x_comp = x_grid_il.ravel()
y_comp = y_grid_il.ravel()

vx_comp = vx_s.data.ravel()
vy_comp = vy_s.data.ravel()

errx_comp = vx_err_s.data.ravel()
erry_comp = vy_err_s.data.ravel()

# Load the ASE Time Series - Ice Velocity
# 450 m resolution for point could
# velocity observations.
dvel = xr.open_dataset(os.path.join(MAIN_PATH,
                                    config['velocity_netcdf']))

# We take everything in 2010 as
# is the year close to BedMachine date
year = '2010'
list_vars = ['vx', 'vy', 'err']

keys = []
for var in list_vars:
    keys.append(f'{var}{year}')

vx_ase = dvel[keys[0]]
vy_ase = dvel[keys[1]]
err_ase = dvel[keys[2]]

x_ase = dvel['xaxis']
y_ase = dvel['yaxis']

# Calculating vector magnitude
vel_obs = np.sqrt(vx_ase**2 + vy_ase**2).rename('vel_obs')

# Now we need the error components of the error magnitude vector
# To find them we first we need to compute the vector direction
# I am assuming the dir is the same as the velocity vector
vel_err_dir = np.arctan(vy_ase/vx_ase) #* 180/np.pi not sure about the angle

# err-y component
err_y = err_ase*np.cos(vel_err_dir).rename('err_y')
# err-x component
err_x = err_ase*np.sin(vel_err_dir).rename('err_y')

# We re-compute the error magnitude to check that the
# error array is the same...
# Surprise! it wont be the same as vy and vx have more nans than
# the err vector.. these nans are passed to vel_err_dir...
# we end up with lest valid data points in "err" due to that.
# But is weird that nan's in the velocity data
# can have error estimates?
err_obs_check = np.sqrt(err_x**2 + err_y**2)
err_obs_check = err_obs_check.rename('err2010')

x_ind_ase = np.where((x_ase >= smith_bbox["xmin"]) & (x_ase <= smith_bbox["xmax"]))[0]
y_ind_ase = np.where((y_ase >= smith_bbox["ymin"]) & (y_ase <= smith_bbox["ymax"]))[0]

x_slice = x_ase[x_ind_ase]
y_slice = y_ase[y_ind_ase]

vx_slice = vx_ase[y_ind_ase, x_ind_ase]
vy_slice = vy_ase[y_ind_ase, x_ind_ase]
err_slice = err_obs_check[y_ind_ase, x_ind_ase]
err_y_slice = err_y[y_ind_ase, x_ind_ase].rename('err_y')
err_x_slice = err_x[y_ind_ase, x_ind_ase].rename('err_x')
vel_obs_slice = vel_obs[y_ind_ase, x_ind_ase]

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

bool_list = meshtools.check_if_arrays_have_same_shape(all_data,
                                                  shape_after)

assert all(element==True for element in bool_list)

mask_comp = np.array(vx_comp, dtype=bool)

file_name = os.path.join(MAIN_PATH,
                         config['smith_vel_obs'])

if os.path.exists(file_name):
  os.remove(file_name)
else:
  print("The file does not exist")

with h5py.File(file_name, 'w') as outty:
    data = outty.create_dataset("mask_vel_comp", mask_comp.shape, dtype='f')
    data[:] = mask_comp
    data = outty.create_dataset("u_comp", vx_comp.shape, dtype='f')
    data[:] = vx_comp
    data = outty.create_dataset("u_comp_std", errx_comp.shape, dtype='f')
    data[:] = errx_comp
    data = outty.create_dataset("v_comp", vy_comp.shape, dtype='f')
    data[:] = vy_comp
    data = outty.create_dataset("v_comp_std", erry_comp.shape, dtype='f')
    data[:] = erry_comp
    data = outty.create_dataset("x_comp", x_comp.shape, dtype='f')
    data[:] = x_comp
    data = outty.create_dataset("y_comp", y_comp.shape, dtype='f')
    data[:] = y_comp
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
