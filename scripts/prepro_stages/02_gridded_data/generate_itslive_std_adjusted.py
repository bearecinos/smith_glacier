"""
Crops ITSLive velocity data to the smith glacier domain for a mosaic and cloud
point data distribution. By default this script will always use ITSLive data
and always provide cloud a cloud point data distribution.

Additionally, the STD of each vel component its adjusted with the absolute
difference between MEaSUREs 2014 and ITSLive 2014 data set.

Options for the composite velocity mosaic:
- ITSLIVE only

Options for the cloud point velocity:
- ITSLIVE 2014
- ITSLIVE 2014 with STDvx and STDvy adjusted according to the following:
    `np.maxima(vx_err_s, np.abs(vx_s-vx_mi_s))`
    where `vx_s-vx_mi_s` is the velocity difference between ITslive 2014
    and MEaSUREs 2014 (MEaSUREs was interpolated to the itslive grid).
- It is possible to subsample each cloud data set by selecting the top left
    value of every sub-grid box, which size is determined by the variable input: -step

The code generates one or two .h5 files; if a step is chosen, with the corresponding velocity
file suffix:
e.g. `*_itslive-comp_std-adjusted-cloud_subsample-training_step-1E+1.h5`
e.g. `*_itslive-comp_std-adjusted-cloud_subsample-test_step-1E+1.h5`

If there is no subsampling:
e.g. `*_itslive-comp_std-adjusted-cloud_subsample-none_step-0E+0.h5`

The files contain the following variables stored as tuples
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
import argparse
import xarray as xr
from decimal import Decimal

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-compute_interpolation",
                    action="store_true",
                    help="If true computes the interpolation of MEaSUREs 2014 vel"
                         "file to the ITSLive grid and saves it as a netcdf file for re-use.")
parser.add_argument("-step",
                    type=int, default=10,
                    help="Sub-box size for the subsample")

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

x_grid, y_grid = np.meshgrid(x_s, y_s)

vx_int = vel_tools.interpolate_missing_data(vx_s, x_grid, y_grid)
vy_int = vel_tools.interpolate_missing_data(vy_s, x_grid, y_grid)
stdvx_int = vel_tools.interpolate_missing_data(vx_std_s, x_grid, y_grid)
stdvy_int = vel_tools.interpolate_missing_data(vy_std_s, x_grid, y_grid)

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
composite_dict = {'x_comp': x_grid.ravel(),
                 'y_comp': y_grid.ravel(),
                 'vx_comp': vx_int,
                 'vy_comp': vy_int,
                 'std_vx_comp': stdvx_int,
                 'std_vy_comp': stdvy_int}

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
diff_vx = np.abs(vxc_s - vx_mi_s)
diff_vy = np.abs(vyc_s -vy_mi_s)

vxc_err_its_2014_adjust = vel_tools.create_adjusted_std_maxima(vxc_err_s, diff_vx)
vyc_err_its_2014_adjust = vel_tools.create_adjusted_std_maxima(vyc_err_s, diff_vy)

step = abs(args.step)

# Are we subsampling this one? yes only with step > 0.

if step != 0:
    print('This is happening')
    # Computing our training set of cloud velocities
    vx_trn, x_trn, y_trn = vel_tools.create_subsample(vxc_s, step, return_coords=True)
    vy_trn = vel_tools.create_subsample(vyc_s, step)
    vx_std_trn = vel_tools.create_subsample(vxc_err_its_2014_adjust, step)
    vy_std_trn = vel_tools.create_subsample(vyc_err_its_2014_adjust, step)

    # Computing our TEST set of cloud velocities
    for x, y in zip(x_trn, y_trn):
        vxc_s.loc[dict(x=x, y=y)] = np.nan
        vyc_s.loc[dict(x=x, y=y)] = np.nan
        vxc_err_its_2014_adjust.loc[dict(x=x, y=y)] = np.nan
        vyc_err_its_2014_adjust.loc[dict(x=x, y=y)] = np.nan

    # Dropping the Nans from the TRAINING set
    out_cloud = vel_tools.drop_nan_from_multiple_numpy(x_trn, y_trn,
                                                       vx_trn, vy_trn,
                                                       vx_std_trn, vy_std_trn)

    cloud_dict_training = {'x_cloud': out_cloud.x.values,
                           'y_cloud': out_cloud.y.values,
                           'vx_cloud': out_cloud.vx.values,
                           'vy_cloud': out_cloud.vy.values,
                           'std_vx_cloud': out_cloud.std_vx.values,
                           'std_vy_cloud': out_cloud.std_vy.values}


    # Dropping the nans from the TEST set
    masked_array = np.ma.masked_invalid(vxc_s.data*vxc_err_its_2014_adjust.data)

    out_test = vel_tools.drop_invalid_data_from_several_arrays(vxc_s.x.values,
                                                               vxc_s.y.values,
                                                               vxc_s,
                                                               vyc_s,
                                                               vxc_err_its_2014_adjust,
                                                               vyc_err_its_2014_adjust,
                                                               masked_array)

    cloud_dict_test = {'x_cloud': out_test[0],
                       'y_cloud': out_test[1],
                       'vx_cloud': out_test[2],
                       'vy_cloud': out_test[3],
                       'std_vx_cloud': out_test[4],
                       'std_vy_cloud': out_test[5]}

    # We write the training file first
    file_suffix = 'itslive-comp_std-adjusted-cloud_subsample-training_step-' + \
                  "{:.0E}".format(Decimal(args.step)) + '.h5'

    file_name_training = os.path.join(MAIN_PATH, config['smith_vel_obs'] + file_suffix)

    vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                          cloud_dict=cloud_dict_training,
                                          fpath=file_name_training)

    # We write the test file second
    file_suffix = 'itslive-comp_std-adjusted-cloud_subsample-test_step-' + \
                  "{:.0E}".format(Decimal(args.step)) + '.h5'

    file_name_test = os.path.join(MAIN_PATH, config['smith_vel_obs'] + file_suffix)

    vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                          cloud_dict=cloud_dict_test,
                                          fpath=file_name_test)
else:
    # We write the complete cloud array!

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
    cloud_dict = {'x_cloud': x_nona,
                  'y_cloud': y_nona,
                  'vx_cloud': vx_nona,
                  'vy_cloud':  vy_nona,
                  'std_vx_cloud': stdvx_nona,
                  'std_vy_cloud': stdvy_nona}


    # We write the test file second
    file_suffix = 'itslive-comp_std-adjusted-cloud_subsample-none_step-' + \
                  "{:.0E}".format(Decimal(args.step)) + '.h5'

    file_name = os.path.join(MAIN_PATH, config['smith_vel_obs'] + file_suffix)

    vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                          cloud_dict=cloud_dict,
                                          fpath=file_name)