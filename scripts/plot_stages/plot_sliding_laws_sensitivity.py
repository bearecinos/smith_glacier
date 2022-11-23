import sys
import os
import salem
import pyproj
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from configobj import ConfigObj
from fenics import *
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh

from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-toml_path_i", type=str,
                    default="../run_experiments/run_workflow/data_products_exp_tomls/smith_cloud_itslive.toml",
                    help="pass .toml file")
parser.add_argument("-toml_path_m", type=str,
                    default="../run_experiments/run_workflow/data_products_exp_tomls/smith_cloud_measures.toml",
                    help="pass .toml file")
parser.add_argument('-n_sens', nargs="+", type=int, help="pass n_sens to plot (max 2)")
parser.add_argument("-sub_plot_dir", type=str, default="temp", help="pass sub plot directory to store the plots")
args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

from ficetools import backend, velocity, graphics, utils_funcs

run_golden = args.toml_path_i
run_cornford = args.toml_path_m

n_sens = args.n_sens

# Read each run params
params_budd = conf.ConfigParser(run_golden)
params_corn = conf.ConfigParser(run_cornford)

#Get the n_sens
num_sens = np.arange(0, params_budd.time.num_sens)
print('Get data for Time')
n_zero = num_sens[n_sens[0]]
print(n_zero)

t_sens = np.flip(np.linspace(params_budd.time.run_length, 0, params_budd.time.num_sens))
t_zero = np.round(t_sens[n_sens[0]])
print(t_zero)

valpha_B, vbeta_B = utils_funcs.compute_vertex_for_dQ_dalpha_component(params_budd,
                                                                       n_sen=n_zero,
                                                                       mult_mmatrix=True)
valpha_C, vbeta_C = utils_funcs.compute_vertex_for_dQ_dalpha_component(params_corn,
                                                                       n_sen=n_zero,
                                                                       mult_mmatrix=True)

print('Get data for Time')
n_last = num_sens[n_sens[-1]]
print(n_last)
t_last = np.round(t_sens[n_sens[-1]])
print(t_last)

valpha_B_40, vbeta_B_40 = utils_funcs.compute_vertex_for_dQ_dalpha_component(params_budd,
                                                                             n_sen=n_last,
                                                                             mult_mmatrix=True)
valpha_C_40, vbeta_C_40 = utils_funcs.compute_vertex_for_dQ_dalpha_component(params_corn,
                                                                             n_sen=n_last,
                                                                             mult_mmatrix=True)

# We get rid of negative sensitivities
valpha_C[valpha_C < 0] = 0
valpha_C_40[valpha_C_40 < 0] = 0
valpha_B_40[valpha_B_40 < 0] = 0
valpha_B[valpha_B < 0] = 0

vbeta_C[vbeta_C < 0] = 0
vbeta_C_40[vbeta_C_40 < 0] = 0
vbeta_B_40[vbeta_B_40 < 0] = 0
vbeta_B[vbeta_B < 0] = 0

perc_C_40 = np.percentile(valpha_C_40, 90)
valpha_C_norm = utils_funcs.normalize(valpha_C, percentile=perc_C_40)
valpha_C_40_norm = utils_funcs.normalize(valpha_C_40, percentile=perc_C_40)

perc_B_40 = np.percentile(valpha_B_40, 90)
valpha_B_norm = utils_funcs.normalize(valpha_B, percentile=perc_B_40)
valpha_B_40_norm = utils_funcs.normalize(valpha_B_40, percentile=perc_B_40)

percbeta_C_40 = np.percentile(vbeta_C_40, 90)
vbeta_C_norm = utils_funcs.normalize(vbeta_C, percentile=percbeta_C_40)
vbeta_C_40_norm = utils_funcs.normalize(vbeta_C_40, percentile=percbeta_C_40)

percbeta_B_40 = np.percentile(vbeta_B_40, 90)
vbeta_B_norm = utils_funcs.normalize(vbeta_B, percentile=percbeta_B_40)
vbeta_B_40_norm = utils_funcs.normalize(vbeta_B_40, percentile=percbeta_B_40)

#Read common data and get mesh information
mesh_in = fice_mesh.get_mesh(params_budd)

x = mesh_in.coordinates()[:, 0]
y = mesh_in.coordinates()[:, 1]
t = mesh_in.cells()

trim = tri.Triangulation(x, y, t)

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['axes.titlesize'] = 18

cmap_sen = sns.color_palette("magma", as_cmap=True)

#Read velocity file used for the inversion
configuration = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))
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

# Lets define our salem grid.
# (but we modified things cuz fabi's code only works for the North! TODO: ask fabien)

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

# Now plotting
r=1.2

tick_options = {'axis':'both','which':'both','bottom':False,
     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

# tick_options_mesh = {'axis':'both','which':'both','bottom':False,
#     'top':True,'left':True,'right':False,'labelleft':True, 'labeltop':True, 'labelbottom':False}

fig1 = plt.figure(figsize=(10*r, 10*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.3)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 2.0
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax0.tricontourf(x_n, y_n, t, valpha_B_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(r'$\frac{\delta Q}{\delta \alpha^{2}}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero+1), prop=dict(size=18), frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
minv = 0
maxv = 2.0
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax1.tricontourf(x_n, y_n, t, valpha_B_40_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(r'$\frac{\delta Q}{\delta \alpha^{2}}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_last), prop=dict(size=18), frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
minv = 0
maxv = 2.0
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax2.tricontourf(x_n, y_n, t, valpha_C_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(r'$\frac{\delta Q}{\delta \alpha^{2}}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero+1), prop=dict(size=18), frameon=True, loc='upper right')
ax2.add_artist(n_text)
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)


ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
minv = 0
maxv = 2.0
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax3.tricontourf(x_n, y_n, t, valpha_C_40_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(r'$\frac{\delta Q}{\delta \alpha^{2}}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_last), prop=dict(size=18), frameon=True, loc='upper right')
ax3.add_artist(n_text)
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)

ax0.title.set_text('Weertman–Budd')
ax1.title.set_text('Weertman–Budd')
ax2.title.set_text('Cornford')
ax3.title.set_text('Cornford')

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'sliding_sensitivities_alpha.png'), bbox_inches='tight', dpi=150)

################################# Beta plot ###################################################

fig1 = plt.figure(figsize=(10*r, 10*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.3)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 2.0
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax0.tricontourf(x_n, y_n, t, vbeta_B_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(r'$\frac{\delta Q}{\delta \beta^{2}}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero+1), prop=dict(size=18), frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
minv = 0
maxv = 2.0
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax1.tricontourf(x_n, y_n, t, vbeta_B_40_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(r'$\frac{\delta Q}{\delta \beta^{2}}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_last), prop=dict(size=18), frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
minv = 0
maxv = 2.0
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax2.tricontourf(x_n, y_n, t, vbeta_C_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(r'$\frac{\delta Q}{\delta \beta^{2}}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero+1), prop=dict(size=18), frameon=True, loc='upper right')
ax2.add_artist(n_text)
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)


ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
minv = 0
maxv = 2.0
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax3.tricontourf(x_n, y_n, t, vbeta_C_40_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(r'$\frac{\delta Q}{\delta \beta^{2}}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_last), prop=dict(size=18), frameon=True, loc='upper right')
ax3.add_artist(n_text)
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)

ax0.title.set_text('Weertman–Budd')
ax1.title.set_text('Weertman–Budd')
ax2.title.set_text('Cornford')
ax3.title.set_text('Cornford')

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'sliding_sensitivities_beta.png'), bbox_inches='tight', dpi=150)