"""
Reads data from Frank Pattyn's Temp.mat and Z.mat files, computes B_bar and writes
an output in .h5 format in the same grid as smith_glacier_bedmachine.h5.
Stores the output in fenics_ice ready format.
"""
import os
import sys
import numpy as np
import h5py
from scipy import io
import pandas as pd
import scipy.interpolate as interp
import argparse
from configobj import ConfigObj

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))


# Main directory path
# This needs changing in bow
MAIN_PATH = config['main_path']
sys.path.append(MAIN_PATH)

from ficetools import utils_funcs as utils

# Read data from files

# temperature and horizontal position file
C = io.loadmat(config['temp_pattyn'])
z = io.loadmat(config['temp_pattyn_zeta'])

# Polar stereographic coordinates
xpat = C['X']
ypat = C['Y']

zeta = z['zeta']
# Zeta is Terrain following coordinate. When zeta=0,
# this is the surface. when zeta=1, this is the bottom.
# when zeta=0.25, this is 25 % of the ice thickness from the surface.
# also zeta spacing is IRREGULAR (see below)

#temperature CONVERTED TO C
T = C['temp507'] - 273.15
# rearrange dimensions to be in line with
# python convention -- 1st axis is vertical
T = np.transpose(T,(2,0,1))

print(np.shape(T))
print(np.shape(zeta))
print(zeta)

# Get 3D Aglen
A = utils.paterson(T) * 365. * 24. * 3600.

# Get 3D Bglen -- will have warnings but this should only be for NaN
Bglen = A**(-1/3)

print(np.shape(Bglen))

# IMPORTANT!!!!
#NEED TO FIND OUT THE UNITS EXPECTED BY FENICS_ICE.
#Above will have units Pa s^(1/3), is Pa yr^(1/3) needed???

# NEED TO AVERAGE B IN VERTICAL --
# BUT CANNOT USE np.mean() because of irregular coordinates.
Bbar = np.trapz(Bglen,zeta[0,:],axis=0)

smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -717500.0,
              'ymax': -528500.0}

## Dealing with Frank's Patterson x and y
x_p = np.int32(C['X'][:][0])*1000
y_p = np.int32(C['Y'][:].T[0])*1000

xmin = smith_bbox['xmin']
xmax = smith_bbox['xmax']
ymin = smith_bbox['ymin']
ymax = smith_bbox['ymax']

window = 5.e4

x_inds = np.where((x_p >= xmin-window) & (x_p <= xmax+window))[0]
y_inds = np.where((y_p >= ymin-window) & (y_p <= ymax+window))[0]

x_s = x_p[x_inds]
y_s = y_p[y_inds]


B = pd.DataFrame(Bbar, index=y_p, columns=x_p)
B = B.replace(np.inf, np.nan)

sel = B.loc[y_s, x_s].values

mask = np.zeros(np.shape(sel))
mask[~np.isnan(sel)] = 1.0

x_r, y_r = np.meshgrid(x_s,y_s)

xnn = x_r[~np.isnan(sel)]
ynn = y_r[~np.isnan(sel)]
snn = sel[~np.isnan(sel)]

gd = interp.griddata((xnn,ynn),snn,(x_r,y_r),method='nearest')


smb = 0.38*np.ones(sel.shape)

with h5py.File(os.path.join(MAIN_PATH,
                            'output/02_gridded_data/smith_bglen.h5'), 'w') as outty:

    data = outty.create_dataset("bglen", gd.shape, dtype='f')
    data[:] = gd
    data = outty.create_dataset("bglenmask", gd.shape, dtype='f')
    data[:] = mask
    data = outty.create_dataset("x", x_s.shape, dtype='f')
    data[:] = x_s
    data = outty.create_dataset("y", y_s.shape, dtype='f')
    data[:] = y_s

with h5py.File(os.path.join(MAIN_PATH,
                            'output/02_gridded_data/smith_smb.h5'), 'w') as outty:

    data = outty.create_dataset("smith_smb", smb.shape, dtype='f')
    data[:] = smb
    data = outty.create_dataset("x", x_s.shape, dtype='f')
    data[:] = x_s
    data = outty.create_dataset("y", y_s.shape, dtype='f')
    data[:] = y_s
