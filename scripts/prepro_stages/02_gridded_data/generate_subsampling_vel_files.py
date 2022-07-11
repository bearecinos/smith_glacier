"""
Crops satellite velocities for individual years and composite means
 from different velocity data products to an area extent defined (for now)
  for the smith glacier experiment.

Options for the composite velocity mosaic:
- MEaSUREs
- ITSLIVE

Options for the cloud point velocity (this script will always add cloud point data):
- Measures without filling gaps 2014
- ITSLIVE without filling gaps 2014
- It is possible to subsample each cloud data set by selecting the top left value of
    every sub-grid box, which size is determined by the variable input: -step
- It is possible to enhance the STD velocity error by a factor X,
    by specifying -error_factor as default this is equal to one.

The code generates two .h5 files, with the corresponding velocity
file suffix, depending on what has been chosen as data:
e.g. `*_measures-comp_measures-cloud_subsample-training-step-1E+1_error-factor-1E+0.h5`
e.g. `*_measures-comp_measures-cloud_subsample-test-step-1E+1_error-factor-1E+0.h5`

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
import argparse
import xarray as xr
from decimal import Decimal

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-composite",
                    type=str, default='measures',
                    help="Data product for the composite velocities: itslive or measures")
parser.add_argument("-error_factor",
                    type=float, default=1.0,
                    help="Enlarge error in observation by a factor")
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


#1) Generate first composite velocities and uncertainties
if args.composite == 'itslive':
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

    vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv, error_factor=args.error_factor)

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
          'point data its ITSlive 2014')
    # Opening files with salem slower than rasterio
    # but they end up more organised in xr.DataArrays

    dv = xr.open_dataset(paths_itslive[3])

    vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv,
                                                              error_factor=args.error_factor)

    vx_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox, return_xarray=True)
    vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox, return_xarray=True)
    vx_err_s = vel_tools.crop_velocity_data_to_extend(std_vx, smith_bbox, return_xarray=True)
    vy_err_s = vel_tools.crop_velocity_data_to_extend(std_vy, smith_bbox, return_xarray=True)

    step = args.step

    # Computing our training sets of cloud velocities
    # vx_trn_0 is the upper left
    # vx_trn_m is the middle point of the array

    vx_trn_0_dic, vx_trn_m_dic = vel_tools.create_subsample(vx_s, step, return_coords=True)

    vx_trn_0 = vx_trn_0_dic['v_trn_0']
    x_trn_0 = vx_trn_0_dic['x_trn_0']
    y_trn_0 = vx_trn_0_dic['y_trn_0']

    vx_trn_m = vx_trn_m_dic['v_trn_m']
    x_trn_m = vx_trn_m_dic['x_trn_m']
    y_trn_m = vx_trn_m_dic['y_trn_m']

    vy_trn_0, vy_trn_m = vel_tools.create_subsample(vy_s, step)
    vx_std_trn_0, vx_std_trn_m = vel_tools.create_subsample(vx_err_s, step)
    vy_std_trn_0, vy_std_trn_m = vel_tools.create_subsample(vy_err_s, step)

    # Computing our test set of cloud velocities
    for x_0, y_0 in zip(x_trn_0, y_trn_0):
        vx_s.loc[dict(x=x_0, y=y_0)] = np.nan
        vy_s.loc[dict(x=x_0, y=y_0)] = np.nan
        vx_err_s.loc[dict(x=x_0, y=y_0)] = np.nan
        vy_err_s.loc[dict(x=x_0, y=y_0)] = np.nan

    for x_m, y_m in zip(x_trn_m, y_trn_m):
        vx_s.loc[dict(x=x_m, y=y_m)] = np.nan
        vy_s.loc[dict(x=x_m, y=y_m)] = np.nan
        vx_err_s.loc[dict(x=x_m, y=y_m)] = np.nan
        vy_err_s.loc[dict(x=x_m, y=y_m)] = np.nan

    # Dropping the Nans from the training set
    out_cloud_0 = vel_tools.drop_nan_from_multiple_numpy(x_trn_0, y_trn_0,
                                                         vx_trn_0, vy_trn_0,
                                                         vx_std_trn_0, vy_std_trn_0)

    out_cloud_m = vel_tools.drop_nan_from_multiple_numpy(x_trn_m, y_trn_m,
                                                         vx_trn_m, vy_trn_m,
                                                         vx_std_trn_m, vy_std_trn_m)

    cloud_dict_training_0 = {f'{name:s}_cloud': getattr(out_cloud_0, name).values for name in ['x', 'y',
                                                                                               'vx', 'vy',
                                                                                               'std_vx', 'std_vy']}

    cloud_dict_training_m = {f'{name:s}_cloud': getattr(out_cloud_m, name).values for name in ['x', 'y',
                                                                                               'vx', 'vy',
                                                                                               'std_vx', 'std_vy']}

    # Dropping the nans from the testing set
    masked_array = np.ma.masked_invalid(vx_s.data)

    out_test = vel_tools.drop_invalid_data_from_several_arrays(vx_s.x.values,
                                                               vx_s.y.values,
                                                               vx_s,
                                                               vy_s,
                                                               vx_err_s,
                                                               vy_err_s,
                                                               masked_array)

    cloud_dict_test = {'x_cloud': out_test[0],
                       'y_cloud': out_test[1],
                       'vx_cloud': out_test[2],
                       'vy_cloud': out_test[3],
                       'std_vx_cloud': out_test[4],
                       'std_vy_cloud': out_test[5]}

else:
    print('The velocity product for the composite solution will be MEaSUREs')

    # First load and process MEaSUREs data for storing a composite mean of
    # all velocity components and uncertainty
    path_measures = os.path.join(MAIN_PATH, config['measures_comp'])

    dm = xr.open_dataset(path_measures)

    vx = dm.VX
    vy = dm.VY
    std_vx = dm.STDX * args.error_factor
    std_vy = dm.STDY * args.error_factor

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
    # Ravel all arrays so they can be stored with
    # a tuple shape (values, )
    composite_dict = {'x_comp': x_grid.ravel(),
                      'y_comp': y_grid.ravel(),
                      'vx_comp': vx_int,
                      'vy_comp': vy_int,
                      'std_vx_comp': stdvx_int,
                      'std_vy_comp': stdvy_int}

    print('The velocity product for the cloud '
          'point data its Measures 2013-2014')

    path_measures = os.path.join(MAIN_PATH, config['measures_cloud'])

    dm = xr.open_dataset(path_measures)

    vx = dm.VX
    vy = dm.VY
    std_vx = dm.STDX * args.error_factor
    std_vy = dm.STDY * args.error_factor

    vx_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox, return_xarray=True)
    vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox, return_xarray=True)
    vx_err_s = vel_tools.crop_velocity_data_to_extend(std_vx, smith_bbox, return_xarray=True)
    vy_err_s = vel_tools.crop_velocity_data_to_extend(std_vy, smith_bbox, return_xarray=True)

    step = args.step

    # Computing our training sets of cloud velocities
    # vx_trn_0 is the upper left
    # vx_trn_m is the middle point of the array
    vx_trn_0_dic, vx_trn_m_dic = vel_tools.create_subsample(vx_s, step, return_coords=True)

    vx_trn_0 = vx_trn_0_dic['v_trn_0']
    x_trn_0 = vx_trn_0_dic['x_trn_0']
    y_trn_0 = vx_trn_0_dic['y_trn_0']

    vx_trn_m = vx_trn_m_dic['v_trn_m']
    x_trn_m = vx_trn_m_dic['x_trn_m']
    y_trn_m = vx_trn_m_dic['y_trn_m']

    vy_trn_0, vy_trn_m = vel_tools.create_subsample(vy_s, step)
    vx_std_trn_0, vx_std_trn_m = vel_tools.create_subsample(vx_err_s, step)
    vy_std_trn_0, vy_std_trn_m = vel_tools.create_subsample(vy_err_s, step)

    # Computing our test set of cloud velocities
    for x_0, y_0 in zip(x_trn_0, y_trn_0):
        vx_s.loc[dict(x=x_0, y=y_0)] = np.nan
        vy_s.loc[dict(x=x_0, y=y_0)] = np.nan
        vx_err_s.loc[dict(x=x_0, y=y_0)] = np.nan
        vy_err_s.loc[dict(x=x_0, y=y_0)] = np.nan

    for x_m, y_m in zip(x_trn_m, y_trn_m):
        vx_s.loc[dict(x=x_m, y=y_m)] = np.nan
        vy_s.loc[dict(x=x_m, y=y_m)] = np.nan
        vx_err_s.loc[dict(x=x_m, y=y_m)] = np.nan
        vy_err_s.loc[dict(x=x_m, y=y_m)] = np.nan

    # Dropping the Nans from the training set
    out_cloud_0 = vel_tools.drop_nan_from_multiple_numpy(x_trn_0, y_trn_0,
                                                         vx_trn_0, vy_trn_0,
                                                         vx_std_trn_0, vy_std_trn_0)

    out_cloud_m = vel_tools.drop_nan_from_multiple_numpy(x_trn_m, y_trn_m,
                                                         vx_trn_m, vy_trn_m,
                                                         vx_std_trn_m, vy_std_trn_m)

    cloud_dict_training_0 = {f'{name:s}_cloud': getattr(out_cloud_0, name).values for name in ['x', 'y',
                                                                                               'vx', 'vy',
                                                                                               'std_vx', 'std_vy']}

    cloud_dict_training_m = {f'{name:s}_cloud': getattr(out_cloud_m, name).values for name in ['x', 'y',
                                                                                               'vx', 'vy',
                                                                                               'std_vx', 'std_vy']}

    # Dropping the nans from the testing set
    masked_array = np.ma.masked_invalid(vx_s.data*vx_err_s.data)

    out_test = vel_tools.drop_invalid_data_from_several_arrays(vx_s.x.values,
                                                               vx_s.y.values,
                                                               vx_s,
                                                               vy_s,
                                                               vx_err_s,
                                                               vy_err_s,
                                                               masked_array)

    cloud_dict_test = {'x_cloud': out_test[0],
                       'y_cloud': out_test[1],
                       'vx_cloud': out_test[2],
                       'vy_cloud': out_test[3],
                       'std_vx_cloud': out_test[4],
                       'std_vy_cloud': out_test[5]}


composite = args.composite + '-comp_'
cloud = args.composite + '-cloud_'

file_suffix_test = composite + cloud + 'subsample-test-step-' + "{:.0E}".format(Decimal(step)) + \
                   '_error-factor-' + "{:.0E}".format(Decimal(args.error_factor)) + '.h5'

file_suffix_training_0 = composite + cloud + 'subsample-training-step-zero-' + "{:.0E}".format(Decimal(step)) + \
                       '_error-factor-' + "{:.0E}".format(Decimal(args.error_factor)) + '.h5'

file_suffix_training_m = composite + cloud + 'subsample-training-step-middle-' + "{:.0E}".format(Decimal(step)) + \
                       '_error-factor-' + "{:.0E}".format(Decimal(args.error_factor)) + '.h5'

file_name_test = os.path.join(MAIN_PATH, config['smith_vel_obs']+file_suffix_test)
file_name_training_0 = os.path.join(MAIN_PATH, config['smith_vel_obs']+file_suffix_training_0)
file_name_training_m = os.path.join(MAIN_PATH, config['smith_vel_obs']+file_suffix_training_m)


vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                  cloud_dict=cloud_dict_test,
                                  fpath=file_name_test)

vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                  cloud_dict=cloud_dict_training_0,
                                  fpath=file_name_training_0)

vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                  cloud_dict=cloud_dict_training_m,
                                  fpath=file_name_training_m)