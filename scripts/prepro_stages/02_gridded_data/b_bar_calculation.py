"""
Reads data from Frank Pattyn's Temp.mat and Z.mat files, computes B_bar and writes
an output in .h5 format in the same grid as smith_glacier_bedmachine.h5.
Stores the output in fenics_ice ready format.

TODO: adjust paths so it uses config.ini and check if the order of the final
array is correct !! also calcuate smb constant field
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
from configobj import ConfigObj

# Load configuration file for more order in paths
config = ConfigObj(os.path.expanduser('~/config.ini'))

# Main directory path
# This needs changing in bow
MAIN_PATH = config['main_path']
sys.path.append(MAIN_PATH)

from meshtools import meshtools as meshtools

# Read data from files

# temperature and horizontal position file
C = io.loadmat('/home/brecinos/smith_glacier/input_data/Temp_2013.mat')

# vertical (nondimensional) coordinate
z = io.loadmat('/home/brecinos/smith_glacier/input_data/Zeta.mat')

# Polar stereographic coordinates
xpat = C['X']
ypat = C['Y']

zeta = z['zeta']
# Zeta is Terrain following coordinate. When zeta=0, this is the surface. when zeta=1, this is the bottom.
# when zeta=0.25, this is 25 % of the ice thickness from the surface.
# also zeta spacing is IRREGULAR (see below)

#temperature CONVERTED TO C
T = C['temp507'] - 273.15
# rearrange dimensions to be in line with python convention -- 1st axis is vertical
T = np.transpose(T,(2,0,1))

print(np.shape(T))
print(np.shape(zeta))
print(zeta)

# Get 3D Aglen
A = meshtools.paterson(T) * 365. * 24. * 3600.

# Get 3D Bglen -- will have warnings but this should only be for NaN
Bglen = A**(-1/3)

print(np.shape(Bglen))

# IMPORTANT!!!!
#NEED TO FIND OUT THE UNITS EXPECTED BY FENICS_ICE.
#Above will have units Pa s^(1/3), is Pa yr^(1/3) needed???

# NEED TO AVERAGE B IN VERTICAL --
# BUT CANNOT USE np.mean() because of irregular coordinates.
Bbar = np.trapz(Bglen,zeta[0,:],axis=0)

# We open Joe's file to get the domain we need!
# TODO: this need to be fixed with an extend variable so we dont need to read Joe's output
# Probably will be best to read in bedmachine smith and try to crop the data to that domain?
# But this can be done later ....
file = '/home/brecinos/smith_glacier/input_data/input_run_joe/smith_bglen.h5'
smith_bglen = h5py.File(file, 'r')

bglen = smith_bglen['bglen'][:]

x_sbglen = smith_bglen['x'][:]
y_sbglen = smith_bglen['y'][:]

## Dealing with Frank's Patterson x and y
x_p = np.int32(C['X'][:][0])*1000
y_p = np.int32(C['Y'][:].T[0])*1000

xmin = np.min(x_sbglen)
xmax = np.max(x_sbglen)
ymin = np.min(y_sbglen)
ymax = np.max(y_sbglen)

x_inds = np.where((x_p >= xmin) & (x_p <= xmax))[0]
y_inds = np.where((y_p >= ymin) & (y_p <= ymax))[0]

x_s = x_p[x_inds]
y_s = y_p[y_inds]

assert x_s.all() == x_sbglen.all()
assert y_s.all() == y_sbglen.all()

B = pd.DataFrame(Bbar, index=y_p, columns=x_p)
B = B.replace(np.inf, np.nan)

sel = B.loc[y_s, x_s]

assert sel.shape == bglen.shape

array = ma.masked_where(bglen == 0, bglen)

xx, yy = np.meshgrid(x_sbglen, y_sbglen)

x_zeros = xx[array.mask]
y_zeros = yy[array.mask]

#get only the valid values
x1 = xx[~array.mask]
y1 = yy[~array.mask]

sel_withzero = sel.copy()

for i, j in enumerate(x_zeros):
    print(sel_withzero.shape)
    sel_withzero.loc[y_zeros[i], j]=0

#mask invalid values
array_two = np.ma.masked_invalid(sel_withzero)

#get only the valid values
x_v = xx[~array_two.mask]
y_v = yy[~array_two.mask]


newarr = array_two[~array_two.mask]

GD1 = interp.griddata((x_v, y_v),
                      newarr.ravel(),
                      (xx, yy),
                      method='linear')

smb = 0.38*np.ones(GD1.shape)

with h5py.File(os.path.join(MAIN_PATH,
                            'output/02_gridded_data/smith_smith_bglen'), 'w') as outty:

    data = outty.create_dataset("bglen", GD1.shape, dtype='f')
    data[:] = GD1
    data = outty.create_dataset("x", x_s.shape, dtype='f')
    data[:] = x_s
    data = outty.create_dataset("y", y_s.shape, dtype='f')
    data[:] = y_s

with h5py.File(os.path.join(MAIN_PATH,
                            'output/02_gridded_data/smith_smb'), 'w') as outty:

    data = outty.create_dataset("smith_smb", smb.shape, dtype='f')
    data[:] = smb
    data = outty.create_dataset("x", x_s.shape, dtype='f')
    data[:] = x_s
    data = outty.create_dataset("y", y_s.shape, dtype='f')
    data[:] = y_s