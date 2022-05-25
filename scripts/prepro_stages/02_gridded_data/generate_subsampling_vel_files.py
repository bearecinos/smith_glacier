"""
Crops satellite velocities for individual years and composite means
 from different velocity data products to an area extent defined (for now)
  for the smith glacier experiment.

Options of data products for the composite velocity mosaic:
- MEaSUREs
- ITSLIVE

Options for the cloud point velocity:
- Measures without filling gaps 2013-2014
- ITSLIVE without filling gaps 2011

The code generates a .h5 file, with the corresponding velocity
file suffix, depending on what has been chosen as data:
e.g. `_itslive-comp_cloud.h5`

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
parser.add_argument("-composite",
                    type=str, default='measures',
                    help="Data product for the composite velocities: itslive or measures")
parser.add_argument("-add_cloud_data",
                    action="store_true",
                    help="If this is specify a year for the data is selected "
                         "we dont interpolate nans and add the data as it is "
                         " as cloud point velocities to the .h5 file")
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

    x_cloud = None
    y_cloud = None
    vx_cloud = None
    vy_cloud = None
    vx_err_cloud = None
    vy_err_cloud = None

    x_t = None
    y_t = None
    vx_tests = None
    vy_tests = None
    vx_err_tests = None
    vy_err_tests = None

    if args.add_cloud_data:
        print('The velocity product for the cloud '
              'point data its ITSlive 2014')
        # Opening files with salem slower than rasterio
        # but they end up more organised in xr.DataArrays
        print(paths_itslive[3])

        dv = xr.open_dataset(paths_itslive[3])

        vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv,
                                                                  error_factor=args.error_factor)

        vx_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox, return_xarray=True)
        vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox, return_xarray=True)
        vx_err_s = vel_tools.crop_velocity_data_to_extend(std_vx, smith_bbox, return_xarray=True)
        vy_err_s = vel_tools.crop_velocity_data_to_extend(std_vy, smith_bbox, return_xarray=True)

        step = args.step

        # Computing our training set of cloud velocities
        vx_trn, x_trn, y_trn = vel_tools.create_subsample(vx_s, step, return_coords=True)
        vy_trn = vel_tools.create_subsample(vy_s, step)
        vx_std_trn = vel_tools.create_subsample(vx_err_s, step)
        vy_std_trn = vel_tools.create_subsample(vy_err_s, step)

        # Computing our test set of cloud velocities
        for x, y in zip(x_trn, y_trn):
            vx_s.loc[dict(x=x, y=y)] = np.nan
            vy_s.loc[dict(x=x, y=y)] = np.nan
            vx_err_s.loc[dict(x=x, y=y)] = np.nan
            vy_err_s.loc[dict(x=x, y=y)] = np.nan

        # Dropping the Nans from the training set
        out_cloud = vel_tools.drop_nan_from_multiple_numpy(x_trn, y_trn,
                                                           vx_trn, vy_trn,
                                                           vx_std_trn, vy_std_trn)
        x_cloud = out_cloud.x.values
        y_cloud = out_cloud.y.values
        vx_cloud = out_cloud.vx.values
        vy_cloud = out_cloud.vy.values
        vx_err_cloud = out_cloud.std_vx.values
        vy_err_cloud = out_cloud.std_vy.values

        # Sanity checks!
        assert np.count_nonzero(np.isnan(vx_cloud)) == 0
        assert np.count_nonzero(np.isnan(vy_cloud)) == 0
        assert np.count_nonzero(np.isnan(vx_err_cloud)) == 0
        assert np.count_nonzero(np.isnan(vy_err_cloud)) == 0
        all_data = [x_cloud, y_cloud,
                    vx_cloud, vy_cloud,
                    vx_err_cloud, vy_err_cloud]
        shape_after = x_cloud.shape
        bool_list = vel_tools.check_if_arrays_have_same_shape(all_data,
                                                              shape_after)
        assert all(element == True for element in bool_list)

        # Dropping the nans from the testing set
        masked_array = np.ma.masked_invalid(vx_s.data)

        out_test = vel_tools.drop_invalid_data_from_several_arrays(vx_s.x.values,
                                                                   vx_s.y.values,
                                                                   vx_s,
                                                                   vy_s,
                                                                   vx_err_s,
                                                                   vy_err_s,
                                                                   masked_array)

        x_t = out_test[0]
        y_t = out_test[1]
        vx_tests = out_test[2]
        vy_tests = out_test[3]
        vx_err_tests = out_test[4]
        vy_err_tests = out_test[5]

        # Sanity checks!
        assert np.count_nonzero(np.isnan(vx_tests)) == 0
        assert np.count_nonzero(np.isnan(vy_tests)) == 0
        assert np.count_nonzero(np.isnan(vx_err_tests)) == 0
        assert np.count_nonzero(np.isnan(vy_err_tests)) == 0

        all_data = [x_t, y_t,
                    vx_tests, vy_tests,
                    vx_err_tests, vy_err_tests]

        shape_after = x_t.shape

        bool_list = vel_tools.check_if_arrays_have_same_shape(all_data,
                                                              shape_after)

        assert all(element == True for element in bool_list)

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
    x_comp = x_grid.ravel()
    y_comp = y_grid.ravel()
    vx_comp = vx_int
    vy_comp = vy_int
    errx_comp = stdvx_int
    erry_comp = stdvy_int

    x_cloud = None
    y_cloud = None
    vx_cloud = None
    vy_cloud = None
    vx_err_cloud = None
    vy_err_cloud = None

    x_t = None
    y_t = None
    vx_tests = None
    vy_tests = None
    vx_err_tests = None
    vy_err_tests = None

    if args.add_cloud_data:
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

        # Computing our training set of cloud velocities
        vx_trn, x_trn, y_trn = vel_tools.create_subsample(vx_s, step, return_coords=True)
        vy_trn = vel_tools.create_subsample(vy_s, step)
        vx_std_trn = vel_tools.create_subsample(vx_err_s, step)
        vy_std_trn = vel_tools.create_subsample(vy_err_s, step)

        # Computing our test set of cloud velocities
        for x, y in zip(x_trn, y_trn):
            vx_s.loc[dict(x=x, y=y)] = np.nan
            vy_s.loc[dict(x=x, y=y)] = np.nan
            vx_err_s.loc[dict(x=x, y=y)] = np.nan
            vy_err_s.loc[dict(x=x, y=y)] = np.nan

        # Dropping the Nans from the training set
        out_cloud = vel_tools.drop_nan_from_multiple_numpy(x_trn, y_trn,
                                                           vx_trn, vy_trn,
                                                           vx_std_trn, vy_std_trn)
        x_cloud = out_cloud.x.values
        y_cloud = out_cloud.y.values
        vx_cloud = out_cloud.vx.values
        vy_cloud = out_cloud.vy.values
        vx_err_cloud = out_cloud.std_vx.values
        vy_err_cloud = out_cloud.std_vy.values

        # Sanity checks!
        assert np.count_nonzero(np.isnan(vx_cloud)) == 0
        assert np.count_nonzero(np.isnan(vy_cloud)) == 0
        assert np.count_nonzero(np.isnan(vx_err_cloud)) == 0
        assert np.count_nonzero(np.isnan(vy_err_cloud)) == 0
        all_data = [x_cloud, y_cloud,
                    vx_cloud, vy_cloud,
                    vx_err_cloud, vy_err_cloud]
        shape_after = x_cloud.shape
        bool_list = vel_tools.check_if_arrays_have_same_shape(all_data,
                                                              shape_after)
        assert all(element == True for element in bool_list)

        # Dropping the nans from the testing set
        masked_array = np.ma.masked_invalid(vx_s.data*vx_err_s.data)

        out_test = vel_tools.drop_invalid_data_from_several_arrays(vx_s.x.values,
                                                                   vx_s.y.values,
                                                                   vx_s,
                                                                   vy_s,
                                                                   vx_err_s,
                                                                   vy_err_s,
                                                                   masked_array)

        x_t = out_test[0]
        y_t = out_test[1]
        vx_tests = out_test[2]
        vy_tests = out_test[3]
        vx_err_tests = out_test[4]
        vy_err_tests = out_test[5]

        # Sanity checks!
        assert np.count_nonzero(np.isnan(vx_tests)) == 0
        assert np.count_nonzero(np.isnan(vy_tests)) == 0
        assert np.count_nonzero(np.isnan(vx_err_tests)) == 0
        assert np.count_nonzero(np.isnan(vy_err_tests)) == 0

        all_data = [x_t, y_t,
                    vx_tests, vy_tests,
                    vx_err_tests, vy_err_tests]

        shape_after = x_t.shape

        bool_list = vel_tools.check_if_arrays_have_same_shape(all_data,
                                                              shape_after)

        assert all(element == True for element in bool_list)

mask_comp = np.array(vx_comp, dtype=bool)
print(mask_comp.shape)

composite = args.composite + '-comp'
if args.add_cloud_data:
    file_suffix = composite + 'cloud_subsampled' + "{:.0E}".format(Decimal(args.step)) +'.h5'
else:
    file_suffix = args.composite + '-comp' + "{:.0E}".format(Decimal(args.step)) + '.h5'

file_name = os.path.join(MAIN_PATH, config['smith_vel_obs']+file_suffix)
print(file_name)

file_test = os.path.join(MAIN_PATH, config['smith_vel_obs']+ '_test_data_' + file_suffix)

if os.path.exists(file_name):
  os.remove(file_name)
else:
  print("The file did not exist before so is being created now")

if os.path.exists(file_test):
  os.remove(file_test)
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
    if args.add_cloud_data:
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

with h5py.File(file_test, 'w') as outty:
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
    if args.add_cloud_data:
        data = outty.create_dataset("u_cloud", vx_tests.shape, dtype='f')
        data[:] = vx_tests
        data = outty.create_dataset("u_cloud_std", vx_err_tests.shape, dtype='f')
        data[:] = vx_err_tests
        data = outty.create_dataset("v_cloud", vy_tests.shape, dtype='f')
        data[:] = vy_tests
        data = outty.create_dataset("v_cloud_std", vy_err_tests.shape, dtype='f')
        data[:] = vy_err_tests
        data = outty.create_dataset("x_cloud", x_t.shape, dtype='f')
        data[:] = x_t
        data = outty.create_dataset("y_cloud", y_t.shape, dtype='f')
        data[:] = y_t