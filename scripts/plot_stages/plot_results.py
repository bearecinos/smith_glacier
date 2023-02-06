import sys
import numpy as np
import h5py
import salem
import xarray as xr
import seaborn as sns
import os
from configobj import ConfigObj

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-sub_plot_dir", type=str, default="temp", help="pass sub plot directory to store the plots")
parser.add_argument("-MITgcm_path", type=str, default="../", help="pass MITgcm repo directory")

args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

path_MITgcm = args.MITgcm_path
sys.path.append(path_MITgcm)

from mds import rdmds

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

from ficetools import utils_funcs, graphics, velocity

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

folder='/exports/csce/datastore/geos/groups/geos_iceocean/brecinos/output_smith_paper/07_post_processing_model'
exptfold='_std-original-complete_C0a2-8_L0a-3200_C0b2-28_L0b-1000_'
filename='fenics_ice_output_gridded_500_std-original-complete_C0a2-8_L0a-3200_C0b2-28_L0b-1000_.npz'

files = np.load(folder+'/'+exptfold+'/'+filename)

#['X', 'Y', 'vel_obs', 'vel_model', 'alpha', 'beta', 'bed', 'thick']

x = files['X']
y = files['Y']
m,n = np.shape(x)

vel_model = np.zeros((m,n+1))

vel_model = files['vel_model']

# this reads the output file land_ice.*** into a ndarray
land_ice = '/home/dgoldber/network_links/iceOceanShare/dgoldber/MITgcm_forinput/fenics_streamice/500m_exp/run/land_ice'
q = rdmds(land_ice, 1, rec=(0,1,8,9))
# surface velocities
u = q[0,:,:]
v = q[1,:,:]
# bed velocities (not sure if this should be included)
ub = q[2,:,:]
vb = q[3,:,:]

vel_str = np.sqrt(u**2+v**2)
velb_str = np.sqrt(ub**2+vb**2)

vel_str[np.isnan(vel_model)] = np.nan

import pyproj

#Read velocity file used for the inversion
vel_obs =os.path.join(MAIN_PATH, configuration['measures_cloud'])
dv = xr.open_dataset(vel_obs)

smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -718450.0,
              'ymax': -527000.0}

vx = dv.VX
vy = dv.VY
std_vx = dv.STDX
std_vy = dv.STDY

# Crop velocity data to the Smith Glacier extend
vx_s = velocity.crop_velocity_data_to_extend(vx, smith_bbox, return_xarray=True)
vy_s = velocity.crop_velocity_data_to_extend(vy, smith_bbox, return_xarray=True)
std_vx_s = velocity.crop_velocity_data_to_extend(std_vx, smith_bbox, return_xarray=True)
std_vy_s = velocity.crop_velocity_data_to_extend(std_vy, smith_bbox,return_xarray=True)

vv = (vx_s**2 + vy_s**2)**0.5
std = (std_vx_s**2 + std_vy_s**2)**0.5

# Lets define our salem grid. (but we modified things cuz fabi's code only works for the North! TODO: ask fabien)

proj = pyproj.Proj('EPSG:3413')
y_grid = vx_s.y
x_grid = vx_s.x

dy = abs(y_grid[0] - y_grid[1])
dx = abs(x_grid[0] - x_grid[1])

# Pixel corner
origin_y = y_grid[0] + dy * 0.5
origin_x = x_grid[0] - dx * 0.5

gv = salem.Grid(nxny=(len(x_grid), len(y_grid)), dxdy=(dx, -1*dy), # We use -dy as this is the Sout Hemisphere somehow salem
                   x0y0=(origin_x, origin_y), proj=proj) # is not picking that up!

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['axes.titlesize'] = 20

cmap_vel=sns.diverging_palette(220, 20, as_cmap=True)

# Now plotting
r=1.2

fig1 = plt.figure(figsize=(14.5*r, 5*r))#, constrained_layout=True)
spec = gridspec.GridSpec(1, 3, wspace=0.3, hspace=0.3)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
smap.set_data(np.flipud(vel_str.data))
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(0)
smap.set_vmax(1000)
smap.set_cmap('viridis')
smap.set_extend('both')
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='velocity [m. $yr^{-1}$]')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
smap.set_data(np.flipud(vel_model.data))
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(0)
smap.set_vmax(1000)
smap.set_cmap('viridis')
smap.set_extend('both')
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='velocity [m. $yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
difference = vel_str-vel_model
smap.set_data(np.flipud(difference.data))
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(0)
smap.set_vmax(100)
smap.set_cmap('crest')
smap.set_extend('both')
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='velocity differences [m. $yr^{-1}$]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)

ax0.title.set_text('MITgcm - STREAMICE')
ax1.title.set_text('FEniCS_ice')
ax2.title.set_text('Model differences')
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'streamice_fenics.png'), bbox_inches='tight', dpi=150)