"""
Crops satellite velocities for individual years and composite means
 from different velocity data products to an area extent defined (for now)
  for the smith glacier experiment.

Options for the composite velocity mosaic:
- MEaSUREs
- ITSLIVE

Options for the cloud point velocity:
- Measures without filling gaps 2014
- ITSLIVE without filling gaps 2014
- It is possible to enhance the STD velocity error by a factor X,
    by specifying -error_factor as default this is equal to one.

The code generates a .h5 file, with the corresponding velocity
file suffix, depending on what has been chosen as data:
e.g. `*_itslive-comp_itslive-cloud_error-factor-1E+0`

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
parser.add_argument("-add_cloud_data",
                    action="store_true",
                    help="If this is specify a year for the data is selected "
                         "we dont interpolate nans and add the data as it is "
                         " as cloud point velocities to the .h5 file")
parser.add_argument("-error_factor",
                    type=float, default=1.0,
                    help="Enlarge error in observation by a factor")

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

    assert '_0000.nc' in paths_itslive[2]
    assert '_2014.nc' in paths_itslive[4]

    dv = xr.open_dataset(paths_itslive[2])

    vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv, error_factor=args.error_factor)

    vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox,
                                                            return_coords=True)
    vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox)
    vx_std_s = vel_tools.crop_velocity_data_to_extend(std_vx, smith_bbox)
    vy_std_s = vel_tools.crop_velocity_data_to_extend(std_vy, smith_bbox)

    x_grid, y_grid = np.meshgrid(x_s, y_s)

    array_ma = np.ma.masked_invalid(vx_s)

    # get only the valid values
    x_nona = x_grid[~array_ma.mask].ravel()
    y_nona = y_grid[~array_ma.mask].ravel()
    vx_nona = vx_s[~array_ma.mask].ravel()
    vy_nona = vy_s[~array_ma.mask].ravel()
    stdvx_nona = vx_std_s[~array_ma.mask].ravel()
    stdvy_nona = vy_std_s[~array_ma.mask].ravel()

    # Ravel all arrays so they can be stored with
    # a tuple shape (values, )
    composite_dict = {'x_comp': x_nona.ravel(),
                      'y_comp': y_nona.ravel(),
                      'vx_comp': vx_nona,
                      'vy_comp': vy_nona,
                      'std_vx_comp': stdvx_nona,
                      'std_vy_comp': stdvy_nona}

    cloud_dict = {'x_cloud': None,
                  'y_cloud': None,
                  'vx_cloud': None,
                  'vy_cloud': None,
                  'std_vx_cloud': None,
                  'std_vy_cloud': None}

    if args.add_cloud_data:
        print('The velocity product for the cloud '
              'point data its ITSlive 2014')
        # Opening files with salem slower than rasterio
        # but they end up more organised in xr.DataArrays
        print(paths_itslive[4])
        dv = xr.open_dataset(paths_itslive[4])

        vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv, error_factor=args.error_factor)

        vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx, smith_bbox,
                                                                return_coords=True)
        vy_s = vel_tools.crop_velocity_data_to_extend(vy, smith_bbox)
        vx_err_s = vel_tools.crop_velocity_data_to_extend(std_vx, smith_bbox)
        vy_err_s = vel_tools.crop_velocity_data_to_extend(std_vy, smith_bbox)

        # Mask arrays
        x_grid, y_grid = np.meshgrid(x_s, y_s)

        # array to mask ... a dot product of component and std
        mask_array = vx_s * vx_err_s

        array_ma = np.ma.masked_invalid(mask_array)

        # get only the valid values
        x_nona = x_grid[~array_ma.mask].ravel()
        y_nona = y_grid[~array_ma.mask].ravel()
        vx_nona = vx_s[~array_ma.mask].ravel()
        vy_nona = vy_s[~array_ma.mask].ravel()
        stdvx_nona = vx_err_s[~array_ma.mask].ravel()
        stdvy_nona = vy_err_s[~array_ma.mask].ravel()

        # Ravel all arrays so they can be stored with
        # a tuple shape (values, )
        cloud_dict = {'x_cloud': x_nona,
                      'y_cloud': y_nona,
                      'vx_cloud': vx_nona,
                      'vy_cloud': vy_nona,
                      'std_vx_cloud': stdvx_nona,
                      'std_vy_cloud': stdvy_nona}
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

    x_grid, y_grid = np.meshgrid(x_s, y_s)

    array_ma = np.ma.masked_invalid(vx_s)

    # get only the valid values
    x_nona = x_grid[~array_ma.mask].ravel()
    y_nona = y_grid[~array_ma.mask].ravel()
    vx_nona = vx_s[~array_ma.mask].ravel()
    vy_nona = vy_s[~array_ma.mask].ravel()
    stdvx_nona = std_vx_s[~array_ma.mask].ravel()
    stdvy_nona = std_vy_s[~array_ma.mask].ravel()

    # Ravel all arrays so they can be stored with
    # a tuple shape (values, )
    composite_dict = {'x_comp': x_nona.ravel(),
                      'y_comp': y_nona.ravel(),
                      'vx_comp': vx_nona,
                      'vy_comp': vy_nona,
                      'std_vx_comp': stdvx_nona,
                      'std_vy_comp': stdvy_nona}

    cloud_dict = {'x_cloud': None,
                  'y_cloud': None,
                  'vx_cloud': None,
                  'vy_cloud': None,
                  'std_vx_cloud': None,
                  'std_vy_cloud': None}

    if args.add_cloud_data:
        print('The velocity product for the cloud '
              'point data its Measures 2013-2014')

        path_measures = os.path.join(MAIN_PATH, config['measures_cloud'])

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

        #array to mask ... a dot product of component and std
        mask_array = vx_s*std_vx_s

        array_ma = np.ma.masked_invalid(mask_array)

        # get only the valid values
        x_nona = x_grid[~array_ma.mask].ravel()
        y_nona = y_grid[~array_ma.mask].ravel()
        vx_nona = vx_s[~array_ma.mask].ravel()
        vy_nona = vy_s[~array_ma.mask].ravel()
        stdvx_nona = std_vx_s[~array_ma.mask].ravel()
        stdvy_nona = std_vy_s[~array_ma.mask].ravel()

        # Ravel all arrays so they can be stored with
        # a tuple shape (values, )
        cloud_dict = {'x_cloud': x_nona,
                      'y_cloud': y_nona,
                      'vx_cloud': vx_nona,
                      'vy_cloud': vy_nona,
                      'std_vx_cloud': stdvx_nona,
                      'std_vy_cloud': stdvy_nona}


composite = args.composite + '-comp_'
cloud = args.composite + '-cloud_'

if args.add_cloud_data:
    file_suffix = composite + cloud + 'error-factor-' + "{:.0E}".format(Decimal(args.error_factor)) +'.h5'
else:
    file_suffix = composite + 'no-cloud_' + 'error-factor-' "{:.0E}".format(Decimal(args.error_factor)) + '.h5'

file_name = os.path.join(MAIN_PATH, config['smith_vel_obs']+file_suffix)


vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                  cloud_dict=cloud_dict,
                                  fpath=file_name)