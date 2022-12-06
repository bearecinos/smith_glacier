"""
Plots and calculates the 6-month offset of ITSLIVE
from Jul to Dec 2014
"""
import sys
import os
import salem
import numpy as np
import xarray as xr
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from configobj import ConfigObj
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-sub_plot_dir", type=str, default="temp", help="pass sub plot directory to store the plots")

args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

from ficetools import utils_funcs, graphics, velocity

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

path_measures = os.path.join(MAIN_PATH, configuration['measures_cloud'])
print(path_measures)

fpath_measures_interp = os.path.join(os.path.dirname(os.path.abspath(path_measures)),
                     'measures_in_itslive_grid.nc')
print(fpath_measures_interp)

assert Path(fpath_measures_interp).is_file(), "File not found"

path_itslive = os.path.join(MAIN_PATH,
                                configuration['itslive'])
file_names = os.listdir(path_itslive)
paths_itslive = []
for f in file_names:
    paths_itslive.append(os.path.join(path_itslive, f))

print(paths_itslive)

assert '_2014.nc' in paths_itslive[4]
assert '_2018.nc' in paths_itslive[3]

ds_its_2014 = xr.open_dataset(paths_itslive[4])
ds_its_2018 = xr.open_dataset(paths_itslive[3])
ds_meas = xr.open_dataset(fpath_measures_interp)

smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -718450.0,
              'ymax': -527000.0}

# Crop data to the smith domain
# We start with measures interpolated
vx_mi_s = velocity.crop_velocity_data_to_extend(ds_meas.vx, smith_bbox, return_xarray=True)
vy_mi_s = velocity.crop_velocity_data_to_extend(ds_meas.vy, smith_bbox, return_xarray=True)

vx_2014, vy_2014, std_vx_2014, std_vy_2014 = velocity.process_itslive_netcdf(ds_its_2014)
vx_2018, vy_2018, std_vx_2018, std_vy_2018 = velocity.process_itslive_netcdf(ds_its_2018)

# now itslive 2014
vx_it2014 = velocity.crop_velocity_data_to_extend(vx_2014, smith_bbox, return_xarray=True)
vy_it2014 = velocity.crop_velocity_data_to_extend(vy_2014, smith_bbox, return_xarray=True)

vx_std_it2014 = velocity.crop_velocity_data_to_extend(std_vx_2014, smith_bbox, return_xarray=True)
vy_std_it2014= velocity.crop_velocity_data_to_extend(std_vy_2014, smith_bbox, return_xarray=True)

# now itslive 2018
vx_it2018 = velocity.crop_velocity_data_to_extend(vx_2018, smith_bbox, return_xarray=True)
vy_it2018 = velocity.crop_velocity_data_to_extend(vy_2018, smith_bbox, return_xarray=True)

vx_std_it2018 = velocity.crop_velocity_data_to_extend(std_vx_2018, smith_bbox, return_xarray=True)
vy_std_it2018= velocity.crop_velocity_data_to_extend(std_vy_2018, smith_bbox, return_xarray=True)

# Calculate the offset
toplot = np.sqrt((vx_it2018-vx_it2014)**2+(vy_it2018-vy_it2014)**2)/8.

# Calculate the difference between data sets
toplot_2 = np.sqrt((vx_it2014-vx_mi_s)**2+(vy_it2014-vy_mi_s)**2)

## Get grid for lat and lon from MEaSUREs original data
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


# Now plotting
r=1.2

fig1 = plt.figure(figsize=(10*r, 5*r))#, constrained_layout=True)
spec = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.3)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
smap.set_data(toplot.data)
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(0)
smap.set_vmax(200)
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
smap.set_data(toplot_2.data)
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(0)
smap.set_vmax(200)
smap.set_cmap('viridis')
smap.set_extend('both')
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='velocity [m. $yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

ax0.title.set_text('ITS_LIVE 6 month offset')
ax1.title.set_text('ITS_LIVE - MEaSUREs')

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'offset.png'), bbox_inches='tight', dpi=150)