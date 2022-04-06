"""
Crops ASE 2010 velocities and different composite mosaic's from
 different velocity data products to an area extend defined for now
  for the smith glacier experiment.

Options of data products for the composite velocity mosaic:
- MEaSUREs
- ITSLIVE

Options for the cloud point velocity:
- Only ASE 2010
- ASE 2010 plus extra data from ITSlive 2011 to fill more nans

The code generates a .h5 file, with the corresponding velocity
file suffix, depending on what has been chosen as data:
e.g. `_itslive-comp_ase-itslive-cloud.h5`

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
                    help="If specified we add ASE 2010 velocities "
                         "as cloud point velocities to the .h5 file")
parser.add_argument("-add_extra_data_to_cloud",
                    action="store_true",
                    help="If specified we add extra data to ASE cloud point velocities")
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
else:
    print('The velocity product for the composite solution will be MEaSUREs')

    # First load and process MEaSUREs data for storing a composite mean of
    # all velocity components and uncertainty
    path_measures = os.path.join(MAIN_PATH, config['measures'])

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
    # vx_int = vel_tools.interpolate_missing_data(vx_s, x_grid, y_grid)
    # vy_int = vel_tools.interpolate_missing_data(vy_s, x_grid, y_grid)
    # stdvx_int = vel_tools.interpolate_missing_data(std_vx_s, x_grid, y_grid)
    # stdvy_int = vel_tools.interpolate_missing_data(std_vy_s, x_grid, y_grid)

    # Ravel all arrays so they can be stored with
    # a tuple shape (values, )
    x_comp = x_grid.ravel()
    y_comp = y_grid.ravel()
    vx_comp = vx_s.ravel()
    vy_comp = vy_s.ravel()
    errx_comp = std_vx_s.ravel()
    erry_comp = std_vy_s.ravel()

x_cloud = None
y_cloud = None
vx_cloud = None
vy_cloud = None
vx_err_cloud = None
vy_err_cloud = None
vel_obs_cloud = None

if args.add_cloud_data:
    #2) Now lets focus on cloud data, we load the ASE Time Series - Ice Velocity 450 m
    # resolution for point could velocity observations.
    dase = xr.open_dataset(os.path.join(MAIN_PATH,
                                        config['velocity_ase_series']))

    dase = dase.rename_dims({'ny': 'y','nx': 'x'})

    # We take everything in 2010 as is the year close to BedMachine date
    year = '2010'
    list_vars = ['vx', 'vy', 'err']

    keys = []
    for var in list_vars:
        keys.append(f'{var}{year}')

    vx_ase = dase[keys[0]]
    vy_ase = dase[keys[1]]
    err_ase = dase[keys[2]]

    x_ase = dase['xaxis']
    y_ase = dase['yaxis']

    # Calculating vector magnitude
    vel_ase_obs = np.sqrt(vx_ase**2 + vy_ase**2).rename('vel_obs')

    # Now we need the error components of the error magnitude vector
    # To find them I am assuming the dir is the same as the velocity vector
    vel_err_dir = np.arctan(vy_ase/vx_ase) #* 180/np.pi not sure about the angle

    # err-y component
    err_y = err_ase*np.cos(vel_err_dir).rename('err_y')
    # err-x component
    err_x = err_ase*np.sin(vel_err_dir).rename('err_y')

    err_obs_check = np.sqrt(err_x**2 + err_y**2)
    err_obs_check = err_obs_check.rename('err2010')

    # Assigned coordinates to xarrays
    vx_ase = vx_ase.assign_coords({"x": x_ase, "y": y_ase})
    vy_ase = vy_ase.assign_coords({"x": x_ase, "y": y_ase})
    err_x = err_x.assign_coords({"x": x_ase, "y": y_ase})
    err_y = err_y.assign_coords({"x": x_ase, "y": y_ase})

    vel_ase_obs = vel_ase_obs.assign_coords({"x": x_ase, "y": y_ase})

    # Crop velocity data to the Smith Glacier extend
    vx_slice, xind_ase, yind_ase = vel_tools.crop_velocity_data_to_extend(vx_ase,
                                                                          smith_bbox,
                                                                          return_xarray=True,
                                                                          return_indexes=True)
    vy_slice = vel_tools.crop_velocity_data_to_extend(vy_ase, smith_bbox, return_xarray=True)
    err_y_slice = vel_tools.crop_velocity_data_to_extend(err_y, smith_bbox, return_xarray=True)
    err_x_slice = vel_tools.crop_velocity_data_to_extend(err_x, smith_bbox, return_xarray=True)

    x_slice = x_ase[xind_ase]
    y_slice = y_ase[yind_ase]

    vel_obs_slice = vel_tools.crop_velocity_data_to_extend(vel_ase_obs, smith_bbox,
                                                           return_xarray=True)

    # 3) We get rid of outliers in the data
    y_out, x_out = np.where(vel_obs_slice.data > 5000)
    vel_obs_slice[y_out, x_out] = np.nan
    vx_slice[y_out, x_out] = np.nan
    vy_slice[y_out, x_out] = np.nan
    err_y_slice[y_out, x_out] = np.nan
    err_x_slice[y_out, x_out] = np.nan

    shape_before = vy_slice.shape
    print('Shape before nan drop')
    print(shape_before)

    if args.add_extra_data_to_cloud:
        print('We will add extra data from ITSLIVE 2011 '
              'to ASE 2010 to cover up more nans')

        path_itslive_2011 = os.path.join(MAIN_PATH,
                                 config['itslive_2011'])

        output_itslive = vel_tools.open_and_crop_itslive_data(path_itslive_2011,
                                                              extend=smith_bbox,
                                                              x_to_int=x_slice,
                                                              y_to_int=y_slice)

        vel_live = output_itslive[0].data
        vx_live = output_itslive[1].data
        vy_live = output_itslive[2].data
        vxerr_live = output_itslive[3].data
        vyerr_live = output_itslive[4].data

        # We add non-nan values from itslive to ASE numpy arrays
        mask_nan = np.isnan(vel_obs_slice)

        print('Number of nans before adding new data')
        print(np.count_nonzero(mask_nan))

        vel_final_ase = vel_obs_slice.data
        vel_final_ase[mask_nan] = vel_live[mask_nan]

        vxf_ase = vx_slice.data
        vxf_ase[mask_nan] = vx_live[mask_nan]

        vyf_ase = vy_slice.data
        vyf_ase[mask_nan] = vy_live[mask_nan]

        errxf_ase = err_x_slice.data
        errxf_ase[mask_nan] = vxerr_live[mask_nan]

        erryf_ase = err_y_slice.data
        erryf_ase[mask_nan] = vyerr_live[mask_nan]
    else:
        print('We add nothing new to ASE')
        vel_final_ase = vel_obs_slice.data
        vxf_ase = vx_slice.data
        vyf_ase = vy_slice.data
        errxf_ase = err_x_slice.data
        erryf_ase = err_y_slice.data

    ## 4) We get rid of nan data for all the cloud arrays
    vel_ma = np.ma.masked_invalid(vel_final_ase)
    print('Number of nans after adding new data')
    print(np.count_nonzero(vel_ma.mask))

    output_cloud = vel_tools.drop_invalid_data_from_several_arrays(x_slice,
                                                                   y_slice,
                                                                   vxf_ase,
                                                                   vyf_ase,
                                                                   errxf_ase,
                                                                   erryf_ase,
                                                                   vel_ma)

    x_cloud = output_cloud[0]
    y_cloud = output_cloud[1]
    vx_cloud = output_cloud[2]
    vy_cloud = output_cloud[3]
    vx_err_cloud = output_cloud[4]
    vy_err_cloud = output_cloud[5]
    vel_obs_cloud = output_cloud[6]

mask_comp = np.array(vx_comp, dtype=bool)


composite = args.composite + '-comp_'
if args.add_cloud_data:
    if args.add_extra_data_to_cloud:
        file_suffix = composite + 'ase-istlive-cloud' + '.h5'
    else:
        file_suffix = composite + 'ase-cloud' + '.h5'
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
