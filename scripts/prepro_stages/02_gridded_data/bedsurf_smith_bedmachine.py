"""
Slice data from BedMachine netCDF and store in fenics_ice ready format.

It also applies gaussian filtering, and redefines the ice base where required for hydrostatic
equilibrium.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import netCDF4
import h5py
from scipy.ndimage import gaussian_filter
import argparse
from configobj import ConfigObj


# Main directory path
# This needs changing in bow
MAIN_PATH = os.path.expanduser('~/scratch/smith_glacier/')
sys.path.append(MAIN_PATH)

from meshtools import meshtools as meshtools

# Load configuration file for more order in paths
config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

parser = argparse.ArgumentParser()
parser.add_argument("-sigma", type=float, default=0.0, help="Sigma value for gauss filter - zero means no filtering")
args = parser.parse_args()

gauss_sigma = args.sigma
filt = gauss_sigma > 0.0

bedmac_file = os.path.join(MAIN_PATH, config['bedmachine'])

# Out files
output_path = os.path.join(MAIN_PATH,
                            'output/02_gridded_data')
if not os.path.exists(output_path):
    os.makedirs(output_path)

out_file = Path(os.path.join(MAIN_PATH, config['smith_glacier_bedmachine']))

if filt:
    out_file = out_file.stem + "_filt_" + str(gauss_sigma) + out_file.suffix

rhoi = 917.0
rhow = 1030.0

smith_bbox = {'xmin': -1607500.0,
              'xmax': -1382500.0,
              'ymin': -717500.0,
              'ymax': -528500.0}

indata = netCDF4.Dataset(bedmac_file)

# xx = indata['x']
# yy = indata['y']

bed, xx, yy = meshtools.slice_netcdf(indata, 'bed', smith_bbox, return_transform=False)
surf, _, _ = meshtools.slice_netcdf(indata, 'surface', smith_bbox, return_transform=False)
thick, _, _ = meshtools.slice_netcdf(indata, 'thickness', smith_bbox, return_transform=False)
mask, _, _ = meshtools.slice_netcdf(indata, 'mask', smith_bbox, return_transform=False)

#####################################################################
# Smooth surf, then redefine ice base as min(bed, surf-floatthick)
#####################################################################

# Convenience function for param sweep sigma (1.5)
def filt_and_show(arr, sigma):
    result = gaussian_filter(arr, sigma)
    plt.matshow(result[50:150, 100:200])
    plt.show()

def gaussian_nan(arr, sigma, trunc=4.0):
    """
    Clever approach to gaussian filter w/ nans:
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    """
    arr1 = arr.copy()
    arr1[np.isnan(arr)] = 0.0

    arr2 = np.ones_like(arr)
    arr2[np.isnan(arr)] = 0.0

    arr1_filt = gaussian_filter(arr1, sigma, truncate=trunc)
    arr2_filt = gaussian_filter(arr2, sigma, truncate=trunc)

    result = arr1_filt / arr2_filt
    result[np.isnan(arr)] = np.nan
    return result

if filt:

    # Smooth the surface
    surf_filt = surf.copy()
    surf_filt[mask <= 1] = np.nan  # mask ocean/nunatak
    surf_filt = gaussian_nan(surf_filt, gauss_sigma)  # gauss filter accounting for nans
    #surf_filt[mask <= 1] = 0.0

    assert np.nanmin(surf_filt) >= 0.0

    # Compute flotation thickness (i.e. max thick)
    float_thick = surf_filt * (rhow / (rhow - rhoi))
    #float_thick[surf_filt == 0.0] = 0.0

    # And the ice base (max of flotation base or BedMachine bed)
    base_float = surf_filt - float_thick
    ice_base = np.maximum(base_float, bed)

    thick_mod = surf_filt - ice_base

    thick_mod = np.clip(thick_mod, a_min=0.0, a_max=None)
    thick_mod[np.isnan(thick_mod)] = 0.0

else:

    thick_mod = thick
    surf_filt = surf

with h5py.File(os.path.join(MAIN_PATH,
                            config['smith_glacier_bedmachine']), 'w') as outty:
    data = outty.create_dataset("bed", bed.shape, dtype='f')
    data[:] = bed
    data = outty.create_dataset("thick", thick_mod.shape, dtype='f')
    data[:] = thick_mod
    data = outty.create_dataset("surf", surf_filt.shape, dtype='f')
    data[:] = surf_filt
    data = outty.create_dataset("x", xx.shape, dtype='f')
    data[:] = xx
    data = outty.create_dataset("y", yy.shape, dtype='f')
    data[:] = yy
