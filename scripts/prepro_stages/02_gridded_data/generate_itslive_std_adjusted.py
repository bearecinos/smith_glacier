"""
Crops ITslive velocity data to the smith glacier domain for the mosaic and cloud
point data distribution.
For the Cloud point data we use 2014 and the STD of each vel component its adjusted.

Options of data products for the composite velocity mosaic:
- ITSLIVE only

Options for the cloud point velocity:
- ITSLIVE 2014
- with STDvx and STDvy adjusted according to the following:
    np.maxima(vx_err_s, np.abs(vx_s-vx_mi_s))
    where vx_s-vx_mi_s is the velocity difference
    between ITslive 2014 and Measures 2014 (interpolated to
    the itslive grid).

The code generates a .h5 file, with the corresponding velocity
file suffix,
e.g. `_itslive-comp_cloud_std_adjust.h5`

The file contain the following variables stored as tuples
and containing no np.nan's:
- 'u_cloud', 'u_cloud_std', 'v_cloud', 'v_cloud_std', 'x_cloud', 'y_cloud'
- 'u_obs', 'u_std', 'v_obs', 'v_std', 'x', 'y', 'mask_vel' -> default composite

each variable with shape: (#values, )

@authors: Fenics_ice contributors
"""
import os
import sys
from configobj import ConfigObj
import numpy as np
import h5py
import argparse
import xarray as xr
from decimal import Decimal

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-compute_interpolation",
                    action="store_true",
                    help="If true computes the interpolation of MEaSUREs 2014 vel"
                         "file to the ITSLive grid and saves it as a netcdf file for re-use.")

args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = config['main_path']
sys.path.append(MAIN_PATH)

from ficetools import velocity as vel_tools

# Define the Smith Glacier extent to crop all velocity data to this region
# IMPORTANT .. verify that the extent is always bigger than the mesh!
smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -718450.0,
              'ymax': -527000.0}

print('The velocity product for the composite solution will be ITSlive')
print('This choice is slightly slower '
      'as the files are in a very different format than MEaSUREs')

# First load and process ITSLive data for storing
# a composite mean of all velocity components and uncertainty
path_itslive = os.path.join(MAIN_PATH,
                            config['itslive'])
file_names = os.listdir(path_itslive)

paths_itslive = []
for f in file_names:
    paths_itslive.append(os.path.join(path_itslive, f))

print(paths_itslive)

dv = xr.open_dataset(paths_itslive[0])

vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv)

vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox,
                                                        return_coords=True)
vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox)
vx_std_s = vel_tools.crop_velocity_data_to_extend(std_vx, smith_bbox)
vy_std_s = vel_tools.crop_velocity_data_to_extend(std_vy, smith_bbox)

print('Lets check all has the same shape')
print(vx_s.shape, vy_s.shape)
print(vx_std_s.shape, vy_std_s.shape)
print(y_s.shape)
print(x_s.shape)

x_grid, y_grid = np.meshgrid(x_s, y_s)

vx_int = vel_tools.interpolate_missing_data(vx_s, x_grid, y_grid)
vy_int = vel_tools.interpolate_missing_data(vy_s, x_grid, y_grid)
stdvx_int = vel_tools.interpolate_missing_data(vx_std_s, x_grid, y_grid)
stdvy_int = vel_tools.interpolate_missing_data(vy_std_s, x_grid, y_grid)

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
x_comp = x_grid.ravel()
y_comp = y_grid.ravel()
vx_comp = vx_int
vy_comp = vy_int
errx_comp = stdvx_int
erry_comp = stdvy_int

print('The velocity product for the cloud '
      'point data its ITSlive 2014 with the STD adjusted')
# Opening files with salem slower than rasterio
# but they end up more organised in xr.DataArrays
path_measures = os.path.join(MAIN_PATH, config['measures_cloud'])
dm = xr.open_dataset(path_measures)

vx_m = dm.VX
vy_m = dm.VY
std_vx_m = dm.STDX
std_vy_m = dm.STDY

dv = xr.open_dataset(paths_itslive[3])
vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv)

fpath = os.path.join(os.path.dirname(os.path.abspath(path_measures)),
                     'measures_in_itslive_grid.nc')
#
if args.compute_interpolation:
    vx_mi = vx_m.interp(y=dv.y.values, x=dv.x.values)
    vy_mi = vy_m.interp(y=dv.y.values, x=dv.x.values)

    with vel_tools.ncDataset(fpath, 'w', format='NETCDF4') as nc:

        nc.author = 'B.M Recinos'
        nc.author_info = 'The University of Edinburgh'

        x_dim = nc.createDimension('x', len(vx_mi.x.values)) # latitude axis
        y_dim = nc.createDimension('y', len(vx_mi.y.values))

        v = nc.createVariable('x', 'f4', ('x',))
        v.units = 'm'
        v.long_name = 'x coordinates'
        v[:] = vx_mi.x.values

        v = nc.createVariable('y', 'f4', ('y',))
        v.units = 'm'
        v.long_name = 'y coordinates'
        v[:] = vx_mi.y.values

        v = nc.createVariable('lat', 'f4', ('y','x'))
        v.units = 'degrees'
        v.long_name = 'latitude'
        v[:] = vx_mi.lat.values

        v = nc.createVariable('lon', 'f4', ('y','x'))
        v.units = 'degrees'
        v.long_name = 'longitude'
        v[:] = vx_mi.lon.values

        v = nc.createVariable('vx', 'f4', ('y','x'))
        v.units = 'm/yr'
        v.long_name = 'vx velocity component'
        v[:] = vx_mi.data

        v = nc.createVariable('vy', 'f4', ('y','x'))
        v.units = 'm/yr'
        v.long_name = 'vy velocity component'
        v[:] = vy_mi.data

ds = xr.open_dataset(fpath)

# Crop data to the smith domain
# We start with measures interpolated
vx_mi_s = vel_tools.crop_velocity_data_to_extend(ds.vx, smith_bbox, return_xarray=True)
vy_mi_s = vel_tools.crop_velocity_data_to_extend(ds.vy, smith_bbox, return_xarray=True)

# now itslive
vxc_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox,
                                                       return_xarray=True)
vyc_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox, return_xarray=True)
vxc_err_s = vel_tools.crop_velocity_data_to_extend(std_vx, smith_bbox, return_xarray=True)
vyc_err_s = vel_tools.crop_velocity_data_to_extend(std_vy, smith_bbox, return_xarray=True)

# We adjust the STD
vxc_err_its_2014_adjust = np.maximum(vxc_err_s, np.abs(vxc_s-vx_mi_s))
vyc_err_its_2014_adjust = np.maximum(vyc_err_s, np.abs(vyc_s-vy_mi_s))

# Mask arrays
x_s = vxc_s.x.data
y_s = vxc_s.y.data

x_grid, y_grid = np.meshgrid(x_s, y_s)

# array to mask ... a dot product of component and std
mask_array = vxc_s.data * vxc_err_its_2014_adjust.data

# Remove nan from cloud data
array_ma = np.ma.masked_invalid(mask_array)

# get only the valid values
x_nona = x_grid[~array_ma.mask].ravel()
y_nona = y_grid[~array_ma.mask].ravel()
vx_nona = vxc_s.data[~array_ma.mask].ravel()
vy_nona = vyc_s.data[~array_ma.mask].ravel()
stdvx_nona = vxc_err_its_2014_adjust.data[~array_ma.mask].ravel()
stdvy_nona = vyc_err_its_2014_adjust.data[~array_ma.mask].ravel()

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
x_cloud = x_nona
y_cloud = y_nona
vx_cloud = vx_nona
vy_cloud = vy_nona
vx_err_cloud = stdvx_nona
vy_err_cloud = stdvy_nona

# Sanity checks!
assert np.count_nonzero(np.isnan(vx_nona)) == 0
assert np.count_nonzero(np.isnan(vy_nona)) == 0
assert np.count_nonzero(np.isnan(vx_err_cloud)) == 0
assert np.count_nonzero(np.isnan(vy_err_cloud)) == 0

all_data = [x_cloud, y_cloud,
            vx_cloud, vy_cloud,
            vx_err_cloud, vy_err_cloud]

shape_after = x_cloud.shape

bool_list = vel_tools.check_if_arrays_have_same_shape(all_data,
                                                      shape_after)

assert all(element == True for element in bool_list)

mask_comp = np.array(vx_comp, dtype=bool)
print(mask_comp.shape)

file_suffix = 'itslive' + '-comp-' + 'cloud_std_adjusted' + '.h5'
file_name = os.path.join(MAIN_PATH, config['smith_vel_obs']+file_suffix)
print(file_name)

if os.path.exists(file_name):
  os.remove(file_name)
else:
  print("The file did not exist before so is being created now")

with h5py.File(file_name, 'w') as outty:
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