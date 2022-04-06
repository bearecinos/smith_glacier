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
import salem

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

    vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox,
                                                            return_coords=True)
    vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox)
    vx_err_s = vel_tools.crop_velocity_data_to_extend(vx_err, smith_bbox)
    vy_err_s = vel_tools.crop_velocity_data_to_extend(vy_err, smith_bbox)

    print('Lets check all has the same shape')
    print(vx_s.shape, vy_s.shape)
    print(vx_err_s.shape, vy_err_s.shape)
    print(y_s.shape)
    print(x_s.shape)

    x_grid, y_grid = np.meshgrid(x_s, y_s)

    # Ravel all arrays so they can be stored with
    # a tuple shape (values, )
    x_comp = x_grid.ravel()
    y_comp = y_grid.ravel()

    vx_comp = vx_s.ravel()
    vy_comp = vy_s.ravel()

    errx_comp = vx_err_s.ravel()
    erry_comp = vy_err_s.ravel()

    x_cloud = None
    y_cloud = None
    vx_cloud = None
    vy_cloud = None
    vx_err_cloud = None
    vy_err_cloud = None

    if args.add_cloud_data:
        print('The velocity product for the cloud '
              'point data its ITSlive 2010')
        # Opening files with salem slower than rasterio
        # but they end up more organised in xr.DataArrays
        dvx = salem.open_xr_dataset(paths_itslive[4])
        dvx_err = salem.open_xr_dataset(paths_itslive[5])
        dvy = salem.open_xr_dataset(paths_itslive[6])
        dvy_err = salem.open_xr_dataset(paths_itslive[7])

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

        vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox,
                                                                return_coords=True)
        vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox)
        vx_err_s = vel_tools.crop_velocity_data_to_extend(vx_err, smith_bbox)
        vy_err_s = vel_tools.crop_velocity_data_to_extend(vy_err, smith_bbox)

        print('Lets check all has the same shape')
        print(vx_s.shape, vy_s.shape)
        print(vx_err_s.shape, vy_err_s.shape)
        print(y_s.shape)
        print(x_s.shape)

        x_grid, y_grid = np.meshgrid(x_s, y_s)

        # Ravel all arrays so they can be stored with
        # a tuple shape (values, )
        x_cloud = x_grid.ravel()
        y_cloud = y_grid.ravel()

        vx_cloud = vx_s.ravel()
        vy_cloud = vy_s.ravel()

        vx_err_cloud = vx_err_s.ravel()
        vy_err_cloud = vy_err_s.ravel()

else:
    print('The velocity product for the composite solution will be MEaSUREs')

    # First load and process MEaSUREs data for storing a composite mean of
    # all velocity components and uncertainty
    path_measures = os.path.join(MAIN_PATH, config['measures_comp'])

    dm = xr.open_dataset(path_measures)

    vx = dm.VX
    vy = dm.VY
    std_vx = dm.STDX
    std_vy = dm.STDY

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

    if args.add_cloud_data:
        print('The velocity product for the cloud '
              'point data its Measures 2013-2014')

        path_measures = os.path.join(MAIN_PATH, config['measures_cloud'])

        dm = xr.open_dataset(path_measures)

        vx = dm.VX
        vy = dm.VY
        std_vx = dm.STDX
        std_vy = dm.STDY

        # Crop velocity data to the Smith Glacier extend
        vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox,
                                                                return_coords=True)
        vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox)
        std_vx_s = vel_tools.crop_velocity_data_to_extend(std_vx, smith_bbox)
        std_vy_s = vel_tools.crop_velocity_data_to_extend(std_vy, smith_bbox)

        # Mask arrays and interpolate nan with nearest neighbor
        x_grid, y_grid = np.meshgrid(x_s, y_s)

        # Ravel all arrays so they can be stored with
        # a tuple shape (values, )
        x_cloud = x_grid.ravel()
        y_cloud = y_grid.ravel()
        vx_cloud = vx_s.ravel()
        vy_cloud = vy_s.ravel()
        vx_err_cloud = std_vx_s.ravel()
        vy_err_cloud = std_vy_s.ravel()

mask_comp = np.array(vx_comp, dtype=bool)
print(mask_comp.shape)

composite = args.composite + '-comp_'
if args.add_cloud_data:
    file_suffix = composite + 'cloud' + '.h5'
else:
    file_suffix = args.composite + '-comp' + '.h5'

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
