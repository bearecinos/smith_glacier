import sys
import argparse
import pandas as pd
import numpy as np
import salem
import pyproj
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from configobj import ConfigObj
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
from decimal import Decimal

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['axes.titlesize'] = 18
rcParams['legend.fontsize'] = 14


# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-sub_plot_dir", type=str,
                    default="temp", help="pass sub plot directory to store the plots")

args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)
from ficetools import backend, utils_funcs, graphics

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

run_path_prior = os.path.join(MAIN_PATH,
                              'scripts/run_experiments/run_workflow/itslive_exp_paper_tomls/tomls_C0a2*')

path_tomls_folder = glob.glob(run_path_prior)
paths_tomls_prior = []
paths_tomls_vel = []

for path in path_tomls_folder:
    print(path)
    if len(os.listdir(path)) == 4:
        for file in os.listdir(path):
            paths_tomls_vel.append(os.path.join(path,file))
    else:
        file = os.listdir(path)
        paths_tomls_prior.append(os.path.join(path, file[0]))

paths_tomls_prior.append(paths_tomls_vel[1])

print('Experiments with different priors toml files -------------- ')
print(paths_tomls_prior)
print('Experiments with different vel input toml files -------------- ')
print(paths_tomls_vel)

dp = pd.DataFrame()

for file in paths_tomls_prior:
    df = utils_funcs.get_prior_information_from_toml(file)
    dp = pd.concat([dp, df], axis=0)

# reset the index
dp.reset_index(drop=True, inplace=True)

dv = pd.DataFrame()
for file in paths_tomls_vel:
    df = utils_funcs.get_prior_information_from_toml(file)
    dv = pd.concat([dv, df], axis=0)

dv.reset_index(drop=True, inplace=True)

vel_label = ['1.6 % of data retained \n and original STD',
             '1.6 % of data retained \n and adjusted STD',
             '100 % of data retained \n and adjusted STD',
             '6.3 % of data retained \n and adjusted STD']

dv['label'] = vel_label

toml_config1 = dp.iloc[0].path_to_toml
toml_config2 = dp.iloc[1].path_to_toml
toml_config3 = dp.iloc[2].path_to_toml
toml_config4 = dp.iloc[3].path_to_toml
toml_gold = dp.iloc[4].path_to_toml

qoi_dict_c1 = graphics.get_data_for_sigma_path_from_toml(toml_config1, main_dir_path=Path(MAIN_PATH))
qoi_dict_c2 = graphics.get_data_for_sigma_path_from_toml(toml_config2, main_dir_path=Path(MAIN_PATH))
qoi_dict_c3 = graphics.get_data_for_sigma_path_from_toml(toml_config3, main_dir_path=Path(MAIN_PATH))
qoi_dict_c4 = graphics.get_data_for_sigma_path_from_toml(toml_config4, main_dir_path=Path(MAIN_PATH))
qoi_dict_gold = graphics.get_data_for_sigma_path_from_toml(toml_gold, main_dir_path=Path(MAIN_PATH))

sigma_conv_c1 = graphics.get_data_for_sigma_convergence_from_toml(toml_config1,
                                                                  main_dir_path=Path(MAIN_PATH))
sigma_conv_c2 = graphics.get_data_for_sigma_convergence_from_toml(toml_config2,
                                                                  main_dir_path=Path(MAIN_PATH))
sigma_conv_c3 = graphics.get_data_for_sigma_convergence_from_toml(toml_config3,
                                                                  main_dir_path=Path(MAIN_PATH))
sigma_conv_c4 = graphics.get_data_for_sigma_convergence_from_toml(toml_config4,
                                                                  main_dir_path=Path(MAIN_PATH))
sigma_conv_gold = graphics.get_data_for_sigma_convergence_from_toml(toml_gold,
                                                                    main_dir_path=Path(MAIN_PATH))

#### Load data for vel files ##########################################################################

toml_config6 = dv.iloc[0].path_to_toml
toml_config7  = dv.iloc[1].path_to_toml
toml_config8 = dv.iloc[2].path_to_toml
print(toml_config8)
print(toml_config7)
print(toml_config6)
toml_config9 = dv.iloc[3].path_to_toml

qoi_dict_c6 = graphics.get_data_for_sigma_path_from_toml(toml_config6, main_dir_path=Path(MAIN_PATH))
qoi_dict_c7 = graphics.get_data_for_sigma_path_from_toml(toml_config7, main_dir_path=Path(MAIN_PATH))
qoi_dict_c8 = graphics.get_data_for_sigma_path_from_toml(toml_config8, main_dir_path=Path(MAIN_PATH))
qoi_dict_c9 = graphics.get_data_for_sigma_path_from_toml(toml_config9, main_dir_path=Path(MAIN_PATH))

sigma_conv_c6 = graphics.get_data_for_sigma_convergence_from_toml(toml_config6,
                                                                  main_dir_path=Path(MAIN_PATH))
sigma_conv_c7 = graphics.get_data_for_sigma_convergence_from_toml(toml_config7,
                                                                  main_dir_path=Path(MAIN_PATH))
sigma_conv_c8 = graphics.get_data_for_sigma_convergence_from_toml(toml_config8,
                                                                  main_dir_path=Path(MAIN_PATH))
sigma_conv_c9 = graphics.get_data_for_sigma_convergence_from_toml(toml_config9,
                                                                  main_dir_path=Path(MAIN_PATH))

#### Save the data frames of configs #############

dp.to_csv(os.path.join(plot_path, 'prior_configs.csv'))
dv.to_csv(os.path.join(plot_path, 'vel_configs.csv'))

g=1.2
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec

fig1 = plt.figure(figsize=(10*g, 10*g))
spec = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.25)

colors = sns.color_palette()
color_prior = sns.color_palette("rocket_r")

####################################### axes 0 ###########################################################
ax0 = plt.subplot(spec[1])
# Plot in the order of prior strength
p2_prior, = ax0.semilogy(qoi_dict_c2['x'],
                         qoi_dict_c2['sigma_prior'],
                         linewidth=3, color=color_prior[0], linestyle='dashed', label='')
p2_post, = ax0.semilogy(qoi_dict_c2['x'],
                        qoi_dict_c2['sigma_post'],
                        color=color_prior[0], label='', linewidth=3)

p5_prior, = ax0.semilogy(qoi_dict_gold['x'],
                         qoi_dict_gold['sigma_prior'],
                         linewidth=3, color=color_prior[1], linestyle='dashed', label='')
p5_post, = ax0.semilogy(qoi_dict_gold['x'],
                        qoi_dict_gold['sigma_post'],
                        linewidth=3, color=color_prior[1], label='')

p4_prior, = ax0.semilogy(qoi_dict_c4['x'],
                         qoi_dict_c4['sigma_prior'],
                         linewidth=3, color=color_prior[2], linestyle='dashed', label='')
p4_post, = ax0.semilogy(qoi_dict_c4['x'],
                        qoi_dict_c4['sigma_post'],
                        linewidth=3, color=color_prior[2], label='')

p1_prior, = ax0.semilogy(qoi_dict_c1['x'],
                         qoi_dict_c1['sigma_prior'],
                         color=color_prior[3], linestyle='dashed', label='', linewidth=3)
p1_post, = ax0.semilogy(qoi_dict_c1['x'],
                        qoi_dict_c1['sigma_post'],
                        color=color_prior[3], label='', linewidth=3)

p3_prior, = ax0.semilogy(qoi_dict_c3['x'],
                         qoi_dict_c3['sigma_prior'],
                         color=color_prior[4], linestyle='dashed', label='', linewidth=3)
p3_post, = ax0.semilogy(qoi_dict_c3['x'], qoi_dict_c3['sigma_post'],
                        color=color_prior[4], label='', linewidth=3)

ax0.grid(True, which="both", ls="-")
ax0.set_xlabel('Time [yrs]')
ax0.set_ylabel(r'$2\sigma$ $Q_{T}$ [$m^3$]')

plt.legend(handles = [p2_prior, p5_prior, p4_prior, p1_prior, p3_prior],
           labels = ['Weak prior',
                     '',
                     '',
                     '',
                     'Strong prior'],frameon=True, fontsize=15)

[xmin01, xmax01, ymin01, ymax01] = ax0.axis()
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

####################################### axes 1 ###########################################################

ax1 = plt.subplot(spec[0])
vel_color = sns.color_palette("mako")
# Oder in terms of data density

p8_prior, = ax1.semilogy(qoi_dict_c8['x'],
                         qoi_dict_c8['sigma_prior'],
                         linewidth=3, color=vel_color[0], linestyle='dashed', label=dv.iloc[2].label)
p8_post, = ax1.semilogy(qoi_dict_c8['x'],
                        qoi_dict_c8['sigma_post'],
                        linewidth=3, color=vel_color[0], label='')

p7_prior, = ax1.semilogy(qoi_dict_c7['x'],
                         qoi_dict_c7['sigma_prior'],
                         linewidth=3, color=vel_color[1], linestyle='dashed', label=dv.iloc[1].label)

p7_post, = ax1.semilogy(qoi_dict_c7['x'],
                        qoi_dict_c7['sigma_post'],
                        linewidth=3, color=vel_color[1], label='')
p6_prior, = ax1.semilogy(qoi_dict_c6['x'],
                         qoi_dict_c6['sigma_prior'],
                         linewidth=3, color=vel_color[2], linestyle='dashed', label=dv.iloc[0].label)
p6_post, = ax1.semilogy(qoi_dict_c6['x'],
                        qoi_dict_c6['sigma_post'], linewidth=3, color=vel_color[2], label='')
ax1.set_ylim(bottom=ymin01, top=ymax01)
ax1.grid(True, which="both", ls="-")
ax1.set_xlabel('Time [yrs]')
ax1.set_ylabel(r'$2\sigma$ $Q_{T}$ [$m^3$]')
plt.legend(loc='lower right', ncol=1,
            borderaxespad=0, frameon=True, fontsize=15)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

####################################### axes 2 ###########################################################
ax2 = plt.subplot(spec[3])
ax2.semilogy(sigma_conv_c2['ind'],
             np.abs(np.diff(sigma_conv_c2['sig']))/np.diff(sigma_conv_c2['eignum']),
             linewidth=1.5, color=color_prior[0])
ax2.plot(sigma_conv_c2['ind2'],
         np.exp(sigma_conv_c2['slope'] * sigma_conv_c2['ind2'] + sigma_conv_c2['inter']),
         linewidth=3, color=color_prior[0], alpha=0.5,
         label=r'$2\sigma^{est}_{full}$ =' + "{:.1E}".format(Decimal(sigma_conv_c2['sigma_full'])) +
               r' $r^2$=' + str(round(sigma_conv_c2['result'].rvalue**2, 3)))

ax2.semilogy(sigma_conv_gold['ind'],
             np.abs(np.diff(sigma_conv_gold['sig']))/np.diff(sigma_conv_gold['eignum']),
             linewidth=1.5, color=color_prior[1])
ax2.plot(sigma_conv_gold['ind2'],
         np.exp(sigma_conv_gold['slope'] * sigma_conv_gold['ind2'] + sigma_conv_gold['inter']),
         linewidth=3, color=color_prior[1], alpha=0.5)
         # label=r'$\sigma^{est}_{full}$ =' + "{:.2e}".format(str(round(sigma_conv_gold['sigma_full'], 5))) +
         #       r' $r^2$=' + str(round(sigma_conv_gold['result'].rvalue**2, 3)))

ax2.semilogy(sigma_conv_c4['ind'],
             np.abs(np.diff(sigma_conv_c4['sig']))/np.diff(sigma_conv_c4['eignum']),
             linewidth=1.5, color=color_prior[2])
ax2.plot(sigma_conv_c4['ind2'],
         np.exp(sigma_conv_c4['slope'] * sigma_conv_c4['ind2'] + sigma_conv_c4['inter']),
         linewidth=3, color=color_prior[2], alpha=0.5)
         # label=r'$\sigma^{est}_{full}$ =' + "{:.1E}".format(Decimal(sigma_conv_c4['sigma_full'])) +
         #        r' $r^2$=' + str(round(sigma_conv_c4['result'].rvalue**2, 3)))

ax2.semilogy(sigma_conv_c1['ind'],
             np.abs(np.diff(sigma_conv_c1['sig']))/np.diff(sigma_conv_c1['eignum']),
             color=color_prior[3], linewidth=1.5)
ax2.plot(sigma_conv_c1['ind2'],
         np.exp(sigma_conv_c1['slope'] * sigma_conv_c1['ind2'] + sigma_conv_c1['inter']),
         color=color_prior[3], alpha=0.5, linewidth=3)
         # label=r'$\sigma^{est}_{full}$ =' + "{:.1E}".format(Decimal(sigma_conv_c1['sigma_full'])) +
         #       r' $r^2$=' + str(round(sigma_conv_c1['result'].rvalue**2, 3)))

ax2.semilogy(sigma_conv_c3['ind'],
             np.abs(np.diff(sigma_conv_c3['sig']))/np.diff(sigma_conv_c3['eignum']),
             linewidth=1.5, color=color_prior[4])
ax2.plot(sigma_conv_c3['ind2'],
         np.exp(sigma_conv_c3['slope'] * sigma_conv_c3['ind2'] + sigma_conv_c3['inter']),
         linewidth=3, color=color_prior[4], alpha=0.5,
         label=r'$2\sigma^{est}_{full}$ =' + "{:.1E}".format(Decimal(sigma_conv_c3['sigma_full'])) +
               r' $r^2$=' + str(round(sigma_conv_c3['result'].rvalue**2, 3)))

[xmin01, xmax01, ymin01, ymax01] = ax2.axis()
ax2.grid(True, which="both", ls="-")
plt.legend(loc='lower left', ncol=1,
            borderaxespad=0, frameon=True, fontsize=15)
ax2.set_xlabel('Eigenvalue index')
ax2.set_ylabel(r'$\delta$ $2\sigma$ $Q_{T}$ [$m^3$]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)

####################################### axes 3 ###########################################################
ax3 = plt.subplot(spec[2])

ax3.semilogy(sigma_conv_c8['ind'],
             np.abs(np.diff(sigma_conv_c8['sig']))/np.diff(sigma_conv_c8['eignum']),
             linewidth=1.5, color=vel_color[0])
ax3.plot(sigma_conv_c8['ind2'],
         np.exp(sigma_conv_c8['slope'] * sigma_conv_c8['ind2'] + sigma_conv_c8['inter']),
         linewidth=3, color=vel_color[0], alpha=0.5,
         label=r'$2\sigma^{est}_{full}$ =' + "{:.1E}".format(Decimal(sigma_conv_c8['sigma_full'])) +
               r' $r^2$=' + str(round(sigma_conv_c8['result'].rvalue**2, 3)))

ax3.semilogy(sigma_conv_c7['ind'],
             np.abs(np.diff(sigma_conv_c7['sig']))/np.diff(sigma_conv_c7['eignum']),
             linewidth=1.5, color=vel_color[1])
ax3.plot(sigma_conv_c7['ind2'],
         np.exp(sigma_conv_c7['slope'] * sigma_conv_c7['ind2'] + sigma_conv_c7['inter']),
         linewidth=3, color=vel_color[1], alpha=0.5,
         label=r'$2\sigma^{est}_{full}$ =' + "{:.1E}".format(Decimal(sigma_conv_c7['sigma_full'])) +
               r' $r^2$=' + str(round(sigma_conv_c7['result'].rvalue**2, 3)))

ax3.semilogy(sigma_conv_c6['ind'],
             np.abs(np.diff(sigma_conv_c6['sig']))/np.diff(sigma_conv_c6['eignum']),
             color=vel_color[2], linewidth=1.5)
ax3.plot(sigma_conv_c6['ind2'],
         np.exp(sigma_conv_c6['slope'] * sigma_conv_c6['ind2'] + sigma_conv_c6['inter']),
         color=vel_color[2], alpha=0.5, linewidth=3,
         label=r'$2\sigma^{est}_{full}$ =' + "{:.1E}".format(Decimal(sigma_conv_c6['sigma_full'])) +
               r' $r^2$=' + str(round(sigma_conv_c6['result'].rvalue**2, 3)))

ax3.grid(True, which="both", ls="-")
ax3.set_ylim(bottom=ymin01, top=ymax01)
ax3.set_xlabel('Eigenvalue index')
ax3.set_ylabel(r'$\delta$ $2\sigma$ $Q_{T}$ [$m^3$]')
plt.legend(loc='lower right', ncol=1,
            borderaxespad=0, frameon=True, fontsize=14)
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'results_Qoi_path_experiments.pdf'))
