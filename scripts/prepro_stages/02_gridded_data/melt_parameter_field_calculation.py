"""
Reads data from a bespoke .mat file of positions in order to construct a mask
of Dotson ice shelf (and possible new shelf) to set differing melt parameters here

"""
import os
import sys
import numpy as np
from pathlib import Path
import netCDF4
import h5py
from scipy import io
import pandas as pd
import scipy.interpolate as interp
import numpy.ma as ma
import argparse
from configobj import ConfigObj
from scipy.interpolate import NearestNDInterpolator
from matplotlib.path import Path

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))


# IMPORTANT: here is where we set our parameters

melt_depth_therm_1 = 600.
melt_depth_therm_2 = 600.
melt_max_1 = 50.
melt_max_2 = 25.

# Main directory path
# This needs changing in bow
MAIN_PATH = config['main_path']
sys.path.append(MAIN_PATH)
output_path = os.path.join(MAIN_PATH,
                            'output/02_gridded_data')
if not os.path.exists(output_path):
    os.makedirs(output_path)


from meshtools import meshtools as meshtools

# grid on which to find mask

smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -717500.0,
              'ymax': -528500.0}
              
xmin = smith_bbox['xmin']
xmax = smith_bbox['xmax']
ymin = smith_bbox['ymin']
ymax = smith_bbox['ymax']

x = np.arange(xmin,xmax+1,1.e3)
y = np.arange(ymin,ymax+1,1.e3)

# read the matlab file

C = io.loadmat(config['dotson_outline']);
xs = C['xdot'].flatten().tolist()
ys = C['ydot'].flatten().tolist()
xgrid,ygrid = np.meshgrid(x,y)
nygrid, nxgrid = np.shape(xgrid)
xgrid, ygrid = xgrid.flatten(), ygrid.flatten()
#poly = zip(xs,ys)
poly = [(xs[i],ys[i]) for i in range(0,len(xs))]
points =np.vstack((xgrid,ygrid)).T
path = Path(poly)
grid = path.contains_points(points).astype(float)
grid = grid + 1
grid = np.reshape(grid,(nygrid,nxgrid))

melt_depth_therm_field = np.zeros(np.shape(grid))
melt_depth_therm_field[grid==1] = melt_depth_therm_1
melt_depth_therm_field[grid==2] = melt_depth_therm_2

melt_max_field = np.zeros(np.shape(grid))
melt_max_field[grid==1] = melt_max_1
melt_max_field[grid==2] = melt_max_2


with h5py.File(os.path.join(output_path,
                            'melt_depth_params.h5'), 'w') as outty:

    data = outty.create_dataset("melt_depth_therm", melt_depth_therm_field.shape, dtype='f')
    data[:] = melt_depth_therm_field
    data = outty.create_dataset("melt_max", melt_max_field.shape, dtype='f')
    data[:] = melt_max_field
    data = outty.create_dataset("x", x.shape, dtype='f')
    data[:] = x
    data = outty.create_dataset("y", y.shape, dtype='f')
    data[:] = y
