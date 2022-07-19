"""
Interpolates ITslive data to MEaSUREs grid and saves it
as a netcdf for later use.

Compose a test file for the stable inference model stage.

Crops satellite velocities from MEaSUREs for 2014 and composite means
to an area extent defined (for now) for the smith glacier experiment.

Options for the composite velocity mosaic:
- MEaSUREs only!

Options for the cloud point velocity data:
- Measures 2014 without filling gaps.
- From this data set we remove the data points where the vector magnitude
of the difference between measures - itslive is > 50 m.
- STDVX and STDVY are set to 1.0

The code generates a .h5 file, with the corresponding velocity
file name suffix:
*_measures-comp_measures-cloud-interpolated-itslive-grid_error-factor-std-one_.h5'

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
import argparse
import xarray as xr
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-compute_interpolation",
                    action="store_true",
                    help="If true computes the interpolation of Itslive 2014 vel"
                         "file to MEaSUREs grid and saves it as a netcdf file for re-use.")
parser.add_argument("-normalise_measures_std",
                    action="store_true",
                    help="If true sets MEaSUREs stdvx and stdvy in cloud data equal to one."
                    )

args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = config['main_path']
sys.path.append(MAIN_PATH)

from ficetools import velocity as vel_tools

smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -718450.0,
              'ymax': -527000.0}

### Define paths for itslive file
path_itslive = os.path.join(MAIN_PATH,
                            config['itslive'])
file_names = os.listdir(path_itslive)

paths_itslive = []
for f in file_names:
    paths_itslive.append(os.path.join(path_itslive, f))
print(paths_itslive)

assert '_0000.nc' in paths_itslive[2]
assert '_2014.nc' in paths_itslive[3]

# Define paths for MEaSUREs
path_measures = os.path.join(MAIN_PATH, config['measures_cloud'])

# Open itslive data
di = xr.open_dataset(paths_itslive[3])

# Open measures cloud data
dm = xr.open_dataset(path_measures)

# where we will store interpolated data
fpath = os.path.join(os.path.dirname(os.path.abspath(path_itslive)),
                     'itslive_in_measures_grid.nc')

################ Interpolation #######################################
if args.compute_interpolation:

    ds = vel_tools.interp_to_measures_grid(di, dm)

    with vel_tools.ncDataset(fpath, 'w', format='NETCDF4') as nc:
        nc.author = 'B.M Recinos'
        nc.author_info = 'The University of Edinburgh'

        x_dim = nc.createDimension('x', len(ds.x.values))  # latitude axis
        y_dim = nc.createDimension('y', len(ds.y.values))

        v = nc.createVariable('x', 'f4', ('x',))
        v.units = 'm'
        v.long_name = 'x coordinates'
        v[:] = ds.x.values

        v = nc.createVariable('y', 'f4', ('y',))
        v.units = 'm'
        v.long_name = 'y coordinates'
        v[:] = ds.y.values

        v = nc.createVariable('vx', 'f4', ('y', 'x'))
        v.units = 'm/yr'
        v.long_name = 'vx velocity component'
        v[:] = ds.vx.data

        v = nc.createVariable('vy', 'f4', ('y', 'x'))
        v.units = 'm/yr'
        v.long_name = 'vy velocity component'
        v[:] = ds.vy.data

        v = nc.createVariable('vx_err', 'f4', ('y', 'x'))
        v.units = 'm/yr'
        v.long_name = 'vx std velocity component'
        v[:] = ds.vx_err.data

        v = nc.createVariable('vy_err', 'f4', ('y', 'x'))
        v.units = 'm/yr'
        v.long_name = 'vy std velocity component'
        v[:] = ds.vy_err.data

##################### Composite part of the file ######################

print('The velocity product for the composite solution will be MEaSUREs')

# First load and process MEaSUREs data for storing a composite mean of
# all velocity components and uncertainty
path_measures_comp = os.path.join(MAIN_PATH, config['measures_comp'])

dm_comp = xr.open_dataset(path_measures_comp)

vx = dm_comp.VX
vy = dm_comp.VY
std_vx = dm_comp.STDX
std_vy = dm_comp.STDY

# Crop velocity data to the Smith Glacier extend
vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox,
                                                        return_coords=True)
vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox)
std_vx_s = vel_tools.crop_velocity_data_to_extend(std_vx, smith_bbox)
std_vy_s = vel_tools.crop_velocity_data_to_extend(std_vy, smith_bbox)

# Mask arrays and interpolate nan with nearest neighbor
x_grid, y_grid = np.meshgrid(x_s, y_s)
vx_int = vel_tools.interpolate_missing_data(vx_s, x_grid, y_grid)
vy_int = vel_tools.interpolate_missing_data(vy_s, x_grid, y_grid)
stdvx_int = vel_tools.interpolate_missing_data(std_vx_s, x_grid, y_grid)
stdvy_int = vel_tools.interpolate_missing_data(std_vy_s, x_grid, y_grid)

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
composite_dict = {'x_comp': x_grid.ravel(),
                  'y_comp': y_grid.ravel(),
                  'vx_comp': vx_int,
                  'vy_comp': vy_int,
                  'std_vx_comp': stdvx_int,
                  'std_vy_comp': stdvy_int}

######################## Cloud part of the file ##########################

# Now we read the file of itslive interpolated to measures data grid
ds = xr.open_dataset(fpath)

vx_i = ds.vx
vy_i = ds.vy

# We get variables in Measures cloud data file

vx_m = dm.VX
vy_m = dm.VY
std_vx_m = dm.STDX
std_vy_m = dm.STDY

# Crop velocity data from MEaSUREs to the Smith Glacier extend
vxm_s = vel_tools.crop_velocity_data_to_extend(vx_m, smith_bbox, return_xarray=True)
vym_s = vel_tools.crop_velocity_data_to_extend(vy_m, smith_bbox, return_xarray=True)
stdm_vx_s = vel_tools.crop_velocity_data_to_extend(std_vx_m, smith_bbox, return_xarray=True)
stdm_vy_s = vel_tools.crop_velocity_data_to_extend(std_vy_m, smith_bbox, return_xarray=True)

# Crop velocity data from Itslive interpolated to the Smith Glacier extend
vxi_s = vel_tools.crop_velocity_data_to_extend(vx_i, smith_bbox, return_xarray=True)
vyi_s = vel_tools.crop_velocity_data_to_extend(vy_i, smith_bbox, return_xarray=True)

# Lets remove the data from MEaSUREs where the vector magnitude
# difference between measures - itslive is > 50 m.
# Calculate vector magnitude
vv_m = (vxm_s**2 + vym_s**2)**0.5
vv_i = (vxi_s**2 + vyi_s**2)**0.5

# Lets make a copy of the original components
meas_vx = vxm_s.copy()
meas_vy = vym_s.copy()

meas_stdvx = stdm_vx_s.copy()
meas_stdvy = stdm_vy_s.copy()

y_ep, x_ep = np.where(np.abs(vv_m-vv_i) >= 50)

x_coord = meas_vx.x[x_ep]
y_coord = meas_vx.y[y_ep]

# Lets selec the data
for x, y in zip(x_coord, y_coord):
    meas_vx.loc[dict(x=x, y=y)] = np.nan
    meas_vy.loc[dict(x=x, y=y)] = np.nan
    meas_stdvx.loc[dict(x=x, y=y)] = np.nan
    meas_stdvy.loc[dict(x=x, y=y)] = np.nan

assert meas_vx.shape == meas_vy.shape
assert meas_stdvx.shape == meas_vy.shape
assert meas_stdvx.shape == meas_stdvy.shape

# Mask arrays
x_s = meas_vx.x.data
y_s = meas_vx.y.data

x_grid, y_grid = np.meshgrid(x_s, y_s)

# array to mask ... a dot product of component and std
mask_array = meas_vx * meas_stdvy

array_ma = np.ma.masked_invalid(mask_array)

# get only the valid values
x_nona = x_grid[~array_ma.mask].ravel()
y_nona = y_grid[~array_ma.mask].ravel()
vx_nona = meas_vx.data[~array_ma.mask].ravel()
vy_nona = meas_vy.data[~array_ma.mask].ravel()

if args.normalise_measures_std:
    stdvx_nona = np.ones(vx_nona.shape, vx_nona.dtype)
    stdvy_nona = np.ones(vy_nona.shape, vy_nona.dtype)
else:
    stdvx_nona = meas_stdvy.data[~array_ma.mask].ravel()
    stdvy_nona = meas_stdvy.data[~array_ma.mask].ravel()

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
cloud_dict = {'x_cloud': x_nona,
              'y_cloud': y_nona,
              'vx_cloud': vx_nona,
              'vy_cloud': vy_nona,
              'std_vx_cloud': stdvx_nona,
              'std_vy_cloud': stdvy_nona}

composite = 'measures' + '-comp_'
cloud = 'measures' + '-cloud-interpolated-itslive-grid_'

if args.normalise_measures_std:
    file_suffix = composite + cloud + 'error-factor-' + 'std-equal-to-one' +'.h5'
else:
    file_suffix = composite + cloud + 'error-factor-' + 'std-original' + '.h5'

file_name = os.path.join(MAIN_PATH, config['smith_vel_obs']+file_suffix)

vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                  cloud_dict=cloud_dict,
                                  fpath=file_name)
