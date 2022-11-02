import sys
import salem
import pyproj
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from pathlib import Path
from configobj import ConfigObj
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
from fenics import *

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

qoi_dict_c1 = graphics.get_data_for_sigma_path_from_toml(run_golden, main_dir_path=Path(MAIN_PATH))
qoi_dict_c2 = graphics.get_data_for_sigma_path_from_toml(run_cornford, main_dir_path=Path(MAIN_PATH))

sigma_conv_c1 = graphics.get_data_for_sigma_convergence_from_toml(run_golden,
                                                                  main_dir_path=Path(MAIN_PATH))
sigma_conv_c2 = graphics.get_data_for_sigma_convergence_from_toml(run_cornford,
                                                                  main_dir_path=Path(MAIN_PATH))

sigma_params_dict_c1 = graphics.get_params_posterior_std(run_golden, main_dir_path=Path(MAIN_PATH))
sigma_params_dict_c2 = graphics.get_params_posterior_std(run_cornford, main_dir_path=Path(MAIN_PATH))

sigma_beta_c1 = sigma_params_dict_c1['sigma_beta']
sigma_beta_c2 = sigma_params_dict_c2['sigma_beta']

params_c1 = conf.ConfigParser(run_golden)
exp_outdir_inv_c1 = utils_funcs.define_stage_output_paths(params_c1, 'inversion', diagnostics=True)

file_float_c1 = "_".join((params_c1.io.run_name + params_c1.inversion.phase_suffix, 'float.xml'))

exp_outdir_invsigma_c1 = utils_funcs.define_stage_output_paths(params_c1, 'inv_sigma')

file_names_invsigma_c1 = graphics.get_file_names_for_invsigma_plot(params_c1)

path_alpha_c1 = exp_outdir_invsigma_c1 / file_names_invsigma_c1[0]

path_float_c1 = exp_outdir_inv_c1 / file_float_c1

assert path_alpha_c1.is_file()
assert path_float_c1.is_file()

params_c2 = conf.ConfigParser(run_cornford)
exp_outdir_inv_c2 = utils_funcs.define_stage_output_paths(params_c2, 'inversion', diagnostics=True)


file_float_c2 = "_".join((params_c2.io.run_name + params_c2.inversion.phase_suffix, 'float.xml'))

exp_outdir_invsigma_c2 = utils_funcs.define_stage_output_paths(params_c2, 'inv_sigma')

file_names_invsigma_c2 = graphics.get_file_names_for_invsigma_plot(params_c2)

path_alpha_c2 = exp_outdir_invsigma_c2 / file_names_invsigma_c2[0]

path_float_c2 = exp_outdir_inv_c2 / file_float_c2

assert path_alpha_c2.is_file()
assert path_float_c2.is_file()

# Reading mesh
mesh_in = fice_mesh.get_mesh(params_c1)

# Compute the function spaces from the Mesh
M = FunctionSpace(mesh_in, 'DG', 0)

x, y, t = graphics.read_fenics_ice_mesh(mesh_in)

trim = tri.Triangulation(x, y, t)

# Get alpha sigma for Budd
alpha_sigma_c1 = Function(M, str(path_alpha_c1))
alpha_sig_c1 = project(alpha_sigma_c1, M)
sigma_alpha_c1 = alpha_sig_c1.compute_vertex_values(mesh_in)

float_fun_c1 = Function(M, str(path_float_c1))
float_pro_c1 = project(float_fun_c1, M)
float_v_c1 = float_pro_c1.compute_vertex_values(mesh_in)

sigma_alpha_c1[float_v_c1 > 0] = 0

## Now for cornford
alpha_sigma_c2 = Function(M, str(path_alpha_c2))
alpha_sig_c2 = project(alpha_sigma_c2, M)
sigma_alpha_c2 = alpha_sig_c2.compute_vertex_values(mesh_in)

float_fun_c2 = Function(M, str(path_float_c2))
float_pro_c2 = project(float_fun_c2, M)
float_v_c2 = float_pro_c2.compute_vertex_values(mesh_in)

sigma_alpha_c2[float_v_c2 > 0] = 0

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
rcParams['axes.titlesize'] = 18
rcParams['legend.fontsize'] = 14

g=1.2

fig1 = plt.figure(figsize=(14*g, 12*g))
spec = gridspec.GridSpec(2, 3, wspace=0.26, hspace=0.02, width_ratios=[1, 1, 1], height_ratios=[1, 0.5])

colors = sns.color_palette()

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 100
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax0.tricontourf(x_n, y_n, t, sigma_alpha_c1, levels=levels, cmap='viridis', extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap('viridis')
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", ticks=ticks,
                         label='Sliding parameter prior STD' + r'($\alpha$)'  +  '\n [m$^{-1/6}$ yr$^{1/6}$ Pa$^1/3$]')
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
minv = 0
maxv = 1000
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax1.tricontourf(x_n, y_n, t, sigma_alpha_c2, levels=levels, cmap='viridis', extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap('viridis')
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", ticks=ticks,
                         label='Sliding parameter prior STD' + r'($\alpha$)'  +  '\n [m$^{-1/6}$ yr$^{1/6}$ Pa$^{1/3}$]')
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
minv = 0
maxv = 30
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax2.tricontourf(x_n, y_n, sigma_params_dict_c1['t'], sigma_beta_c1, levels=levels,
                    cmap='viridis', extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap('viridis')
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", ticks=ticks,
                         label='Ice stiffness parameter STD' + r'($\beta$)' + '\n [Pa$^{1/2}$. yr$^{1/6}$]')
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
p1, = ax3.plot(qoi_dict_c1['x'], qoi_dict_c1['y'], color=colors[0], label='', linewidth=1)
ax3.fill_between(qoi_dict_c1['x'],
                 qoi_dict_c1['y']-qoi_dict_c1['sigma_post'],
                 qoi_dict_c1['y']+qoi_dict_c1['sigma_post'], facecolor=colors[0], alpha=0.5)

p2, = ax3.plot(qoi_dict_c2['x'], qoi_dict_c2['y'], color=colors[1], label='', linewidth=1)
ax3.fill_between(qoi_dict_c2['x'],
                 qoi_dict_c2['y']-qoi_dict_c2['sigma_post'],
                 qoi_dict_c2['y']+qoi_dict_c2['sigma_post'], facecolor=colors[1], alpha=0.5)

plt.legend(handles = [p1, p2],
           labels = ['Weertman–Budd',
                     'Cornford'],frameon=True, fontsize=14)
ax3.set_ylabel(r'Q$(m^3)$')

at = AnchoredText('d', prop=dict(size=14), frameon=True, loc='upper left')
ax3.add_artist(at)

ax4 = plt.subplot(spec[4])
p1_prior, = ax4.semilogy(qoi_dict_c1['x'], qoi_dict_c1['sigma_prior'],
             color=colors[0], linestyle='dashed', label='', linewidth=3)
p1_post, = ax4.semilogy(qoi_dict_c1['x'], qoi_dict_c1['sigma_post'],
             color=colors[0], label='', linewidth=3)
p2_prior, = ax4.semilogy(qoi_dict_c2['x'], qoi_dict_c2['sigma_prior'], linewidth=3,
             color=colors[1], linestyle='dashed', label='')
p2_post, = ax4.semilogy(qoi_dict_c2['x'], qoi_dict_c2['sigma_post'],
             color=colors[1], label='', linewidth=3)
ax4.grid(True, which="both", ls="-")
ax4.set_xlabel('Time (yrs)')
ax4.set_ylabel(r'$\sigma$ Q$(m^3)$')
plt.legend(handles = [p1_prior, p1_post],
           labels = ['Prior',
                     'Posterior'],frameon=True, fontsize=15)
at = AnchoredText('e', prop=dict(size=14), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
ax5.semilogy(sigma_conv_c1['ind'],
             np.abs(np.diff(sigma_conv_c1['sig']))/np.diff(sigma_conv_c1['eignum']),
             color=colors[0], linewidth=1.5)
ax5.plot(sigma_conv_c1['ind2'],
         np.exp(sigma_conv_c1['slope'] * sigma_conv_c1['ind2'] + sigma_conv_c1['inter']),
         color=colors[0], alpha=0.5, linewidth=3,
        label='slope =' + str(round(sigma_conv_c1['slope'], 5)) + '\n' +r' $R^2$=' + str(round(sigma_conv_c1['result'].rvalue**2, 3)))

ax5.semilogy(sigma_conv_c2['ind'],
             np.abs(np.diff(sigma_conv_c2['sig']))/np.diff(sigma_conv_c2['eignum']), linewidth=1.5,
             color=colors[1])
ax5.plot(sigma_conv_c2['ind2'],
         np.exp(sigma_conv_c2['slope'] * sigma_conv_c2['ind2'] + sigma_conv_c2['inter']), linewidth=3,
         color=colors[1], alpha=0.5,
        label='slope =' + str(round(sigma_conv_c2['slope'], 5)) + '\n' + r' $R^2$=' + str(round(sigma_conv_c2['result'].rvalue**2, 3)))


ax5.grid(True, which="both", ls="-")
plt.legend(loc='upper right', ncol=1,
            borderaxespad=0, frameon=True, fontsize=13)
ax5.set_xlabel('No. of Eigen values')
ax5.set_ylabel(r'$\delta$$\sigma$ Q$(m^3)$')
at = AnchoredText('f', prop=dict(size=14), frameon=True, loc='upper left')
ax5.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'sliding_differences.png'),
            bbox_inches='tight')
