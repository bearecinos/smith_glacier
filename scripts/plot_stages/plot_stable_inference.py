import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import os
import sys
import numpy as np
import argparse
import xarray as xr
from configobj import ConfigObj
from pathlib import Path
import h5py
import pyproj
import salem
from salem import DataLevels
import seaborn as sns

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-output_exp", type=str, default="output/05_stable_inference/output/inversion",
                    help="pass location of stable inference output for multiple tomls")
parser.add_argument("-sub_plot_dir", type=str, default="temp", help="pass sub plot directory to store the plots")

args = parser.parse_args()

config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

from ficetools import utils_funcs, graphics, velocity

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

output_exp = os.path.join(MAIN_PATH, args.output_exp)

input_vel_files = os.path.join(MAIN_PATH, 'input_data/input_run_inv')

# Middle subset
vel_files_m = []
# Upper corner subset
vel_files_0 = []

for root, dirs, files in os.walk(input_vel_files):
    for file in files:
        if file.startswith("smith_obs_vel_itslive-comp_itslive-cloud_subsample-training-step-middle"):
            vel_files_m.append(os.path.join(root, file))

for root, dirs, files in os.walk(input_vel_files):
    for file in files:
        if file.startswith("smith_obs_vel_itslive-comp_itslive-cloud_subsample-training-step-zero"):
            vel_files_0.append(os.path.join(root, file))

# Complete velocity data set
vel_file_all = os.path.join(input_vel_files,
                            'smith_obs_vel_itslive-comp_itslive-cloud_error-factor-1E+0.h5')

shape_x1, shape_y1 = utils_funcs.get_pts_from_h5_velfile(vel_file_all)

step_vels = ['1E+0', ]
n0_pts = [687412, ]

for f in vel_files_0:
    shape_x, shape_y = utils_funcs.get_pts_from_h5_velfile(f)
    f_name = os.path.basename(f)
    step_vf = f_name.split('_')[5].split('-')[-1]
    step_vels.append(step_vf)
    n0_pts = np.append(n0_pts, shape_x)

step_vels = [float(i) for i in step_vels]
n0_pts = (n0_pts*100)/n0_pts[0]

print('Sub set steps corner')
print(step_vels)
print('Number of points corner')
print(n0_pts)

step_vels_m = ['1E+0', ]
n0_pts_m = [687412, ]

for f in vel_files_m:
    shape_x, shape_y = utils_funcs.get_pts_from_h5_velfile(f)
    f_name = os.path.basename(f)
    step_vf = f_name.split('_')[5].split('-')[-1]
    step_vels_m.append(step_vf)
    n0_pts_m = np.append(n0_pts_m, shape_x)

step_vels_m = [float(i) for i in step_vels_m]
n0_pts_m = (n0_pts_m*100)/n0_pts_m[0]


print('Sub set steps middle')
print(step_vels_m)
print('Number of points middle')
print(n0_pts_m)

n0_pts_sorted = -np.sort(-n0_pts)
print('Number of points sorted')
print(n0_pts_sorted)

sub_dirs = []

for rootdir, dirs, files in os.walk(output_exp):
    for subdir in dirs:
        sub_dirs.append(os.path.join(rootdir, subdir))

print(sub_dirs)

sub_dirs.sort(key=lambda f: f in 'subsample')

# Lets make the array a pandas Series
sub_dirss = pd.Series(sub_dirs, index=None)

# Lets read each J_cost output for each inversion in the stable inference test
csv_list = []

for sub in sub_dirss.values:
    for root, dirs, files in os.walk(sub):
        for file in files:
            if file.endswith("_Js.csv"):
                csv_list.append(os.path.join(root, file))

# Lets make a pandas Data frame to store the data from Js_csv
ds = pd.DataFrame()

for file in csv_list:
    # print(file.split('__'))
    name = file.split('__')[-2]

    df = pd.read_csv(file)
    df['experiment'] = name
    df['data_type'] = name.split('_')[0] + '_' + name.split('_')[-1]
    df['step'] = name.split('_')[2]

    ds = pd.concat([ds, df], axis=0)

    # reset the index
    ds.reset_index(drop=True, inplace=True)

ds = ds.astype({'step':'float'})

# Select the data that we need to plot J_cost value against step
to_plot_training_middle_std = ds[ds.data_type == 'subsample_middle'].sort_values(by=['step'],
                                                                                 ascending=True)
to_plot_training_zero_std = ds[ds.data_type == 'subsample_zero'].sort_values(by=['step'],
                                                                             ascending=True)

to_plot_testing_middle_std = ds[ds.data_type == 'testing_middle'].sort_values(by=['step'],
                                                                              ascending=True)
to_plot_testing_zero_std = ds[ds.data_type == 'testing_zero'].sort_values(by=['step'],
                                                                          ascending=True)

color_palette = sns.color_palette("deep")

color_array = [color_palette[0], color_palette[2],
               color_palette[1]]


rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 5
sns.set_context('poster')

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(2, 1, 1)

ax.plot(to_plot_testing_zero_std.step, to_plot_testing_zero_std.J, '-o',
        color=color_array[2],
        markersize=15, lw=3, label='Validating against upper cell points')
ax.plot(to_plot_testing_middle_std.step, to_plot_testing_middle_std.J, '-*',
        color=sns.xkcd_rgb["dark orange"],
        markersize=10, lw=2, label='Validating against middle cell points')

ax.set_ylabel('J')
ax.set_xlim(0,11)
ax.ticklabel_format(axis='y', style='scientific', scilimits=(4,4))
ax.grid()

ax.set_xticks(to_plot_testing_zero_std.step)
ax.set_xticklabels(np.round(n0_pts_sorted, 1))

ax.set_xlabel('% of data points retained from the training set')
ax.legend(fontsize=24)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'stable_inference.png'),
            bbox_inches='tight', dpi=150)

################################################################################################
# We also need to plot some input data for visualization
# In this case we choose the one that retains only 1% of the data
vel_trn = vel_files_0[2]

# Consider putting this in config.ini?
validation_vel_fname = 'smith_obs_vel_measures-comp_measures-cloud-interpolated-itslive-grid_error-factor-std-original.h5'

vel_test = os.path.join(MAIN_PATH,
                         'input_data/input_run_inv/'+validation_vel_fname)

# Read input data for training sets
path_vel_trn = Path(vel_trn)
f_vel_trn = h5py.File(path_vel_trn, 'r')

# Read input data for validation sets
path_vel_test = Path(vel_test)
f_vel_test = h5py.File(path_vel_test, 'r')

# Get data to plot for training set
x_trn = f_vel_trn['x_cloud'][:]
y_trn = f_vel_trn['y_cloud'][:]
u_obs = f_vel_trn['u_cloud'][:]
v_obs = f_vel_trn['v_cloud'][:]

vv_trn = (u_obs**2 + v_obs**2)**0.5

u_obs_std = f_vel_trn['u_cloud_std'][:]
v_obs_std = f_vel_trn['v_cloud_std'][:]

vv_trn_std = (u_obs_std**2 + v_obs_std**2)**0.5

# Get data to plot for validation set
x_val = f_vel_test['x_cloud'][:]
y_val = f_vel_test['y_cloud'][:]
u_obs_val = f_vel_test['u_cloud'][:]
v_obs_val = f_vel_test['v_cloud'][:]

vv_val = (u_obs_val**2 + v_obs_val**2)**0.5

u_obs_std_val = f_vel_test['u_cloud_std'][:]
v_obs_std_val = f_vel_test['v_cloud_std'][:]

vv_val_std = (u_obs_std_val**2 + v_obs_std_val**2)**0.5

# Load grid for lat and lon contours
vel_obs =os.path.join(MAIN_PATH,
                      configuration['measures_cloud'])
dv = xr.open_dataset(vel_obs)

smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -718450.0,
              'ymax': -527000.0}

vx = dv.VX

# Crop velocity data to the Smith Glacier extend
vx_s = velocity.crop_velocity_data_to_extend(vx, smith_bbox, return_xarray=True)

# Lets define our salem grid.
# (but we modified things cuz fabi's code only works for the North!)
# TODO: ask fabien!

proj = pyproj.Proj('EPSG:3413')
y_grid = vx_s.y
x_grid = vx_s.x

dy = abs(y_grid[0] - y_grid[1])
dx = abs(x_grid[0] - x_grid[1])

# Pixel corner
origin_y = y_grid[0] + dy * 0.5
origin_x = x_grid[0] - dx * 0.5

gv = salem.Grid(nxny=(len(x_grid), len(y_grid)),
                dxdy=(dx, -1*dy),
                # We use -dy as this is the South Hemisphere somehow salem is not picking that up!
                x0y0=(origin_x, origin_y),
                proj=proj)

# Now plotting
r= 1.6

tick_options = {'axis':'both','which':'both','bottom':False,
     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

fig1 = plt.figure(figsize=(8*r, 10*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.1)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x_trn, y_trn,
                              crs=gv.proj)

dl = DataLevels(vv_trn, levels=np.arange(0, 1200, 10), extend='both')
c = ax0.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax0, position='bottom', size="5%", pad=0.5, label='velocity [m. $yr^{-1}$]')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
dl = DataLevels(vv_trn_std, levels=np.arange(0, 100, 1), extend='both')
c = ax1.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax1, position='bottom', size="5%", pad=0.5, label='velocity STD [m. $yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
x_n, y_n = smap.grid.transform(x_val, y_val,
                              crs=gv.proj)
dl = DataLevels(vv_val, levels=np.arange(0, 1200, 10), extend='both')
c = ax2.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax2, position='bottom', size="5%", pad=0.5, label='velocity [m. $yr^{-1}$]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
dl = DataLevels(vv_val_std, levels=np.arange(0, 100, 1), extend='both')
c = ax3.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax3, position='bottom', size="5%", pad=0.5, label='velocity STD [m. $yr^{-1}$]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)

ax0.set_title('Training set 1% of ITSLive data set', fontdict={'fontsize': 20})
ax1.set_title('Training set 1% of ITSLive data set', fontdict={'fontsize': 20})
ax2.set_title('Validation set MEaSUREs data set', fontdict={'fontsize': 20})
ax3.set_title('Validation set MEaSUREs data set', fontdict={'fontsize': 20})

plt.tight_layout()
plt.savefig(os.path.join(plot_path,'stable_inference_input.png'),
            bbox_inches='tight',
            dpi=150)
