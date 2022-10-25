import sys
import argparse
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['axes.titlesize'] = 18

from pathlib import Path
from configobj import ConfigObj

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-tomls_one_path", type=str,
                    default="../run_experiments/run_workflow/data_products_exp_tomls/smith_cloud_itslive.toml",
                    help="pass .toml file")
parser.add_argument("-tomls_two_path", type=str,
                    default="../run_experiments/run_workflow/data_products_exp_tomls/smith_cloud_itslive.toml",
                    help="pass .toml file")
parser.add_argument("-tomls_three_path", type=str,
                    default="../run_experiments/run_workflow/data_products_exp_tomls/smith_cloud_itslive.toml",
                    help="pass .toml file")
parser.add_argument("-tomls_four_path", type=str,
                    default="../run_experiments/run_workflow/data_products_exp_tomls/smith_cloud_itslive.toml",
                    help="pass .toml file")
parser.add_argument("-sub_plot_dir", type=str,
                    default="temp", help="pass sub plot directory to store the plots")

args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)
from ficetools import graphics

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

toml_config1 = os.path.join(args.tomls_one_path)
toml_config2 = os.path.join(args.tomls_two_path)
toml_config3 = os.path.join(args.tomls_three_path)
toml_config4 = os.path.join(args.tomls_four_path)

qoi_dict_c1 = graphics.get_data_for_sigma_path_from_toml(toml_config1, main_dir_path=Path(MAIN_PATH))
qoi_dict_c2 = graphics.get_data_for_sigma_path_from_toml(toml_config2, main_dir_path=Path(MAIN_PATH))
qoi_dict_c3 = graphics.get_data_for_sigma_path_from_toml(toml_config3, main_dir_path=Path(MAIN_PATH))
qoi_dict_c4 = graphics.get_data_for_sigma_path_from_toml(toml_config4, main_dir_path=Path(MAIN_PATH))

sigma_params_dict_c1 = graphics.get_params_posterior_std(toml_config1, main_dir_path=Path(MAIN_PATH))
sigma_params_dict_c2 = graphics.get_params_posterior_std(toml_config2, main_dir_path=Path(MAIN_PATH))
sigma_params_dict_c3 = graphics.get_params_posterior_std(toml_config3, main_dir_path=Path(MAIN_PATH))
sigma_params_dict_c4 = graphics.get_params_posterior_std(toml_config4, main_dir_path=Path(MAIN_PATH))

sigma_alpha_c1 = sigma_params_dict_c1['sigma_alpha']
sigma_alpha_c2 = sigma_params_dict_c2['sigma_alpha']
sigma_alpha_c3 = sigma_params_dict_c3['sigma_alpha']
sigma_alpha_c4 = sigma_params_dict_c4['sigma_alpha']

sigma_beta_c1 = sigma_params_dict_c1['sigma_beta']
sigma_beta_c2 = sigma_params_dict_c2['sigma_beta']
sigma_beta_c3 = sigma_params_dict_c3['sigma_beta']
sigma_beta_c4 = sigma_params_dict_c4['sigma_beta']

prior_alpha_c1 = sigma_params_dict_c1['prior_alpha']
prior_alpha_c2 = sigma_params_dict_c2['prior_alpha']
prior_alpha_c3 = sigma_params_dict_c3['prior_alpha']
prior_alpha_c4 = sigma_params_dict_c4['prior_alpha']

prior_beta_c1 = sigma_params_dict_c1['prior_beta']
prior_beta_c2 = sigma_params_dict_c2['prior_beta']
prior_beta_c3 = sigma_params_dict_c3['prior_beta']
prior_beta_c4 = sigma_params_dict_c4['prior_beta']

sigma_conv_c1 = graphics.get_data_for_sigma_convergence_from_toml(toml_config1,
                                                                  main_dir_path=Path(MAIN_PATH),
                                                                  startind=3000)
sigma_conv_c2 = graphics.get_data_for_sigma_convergence_from_toml(toml_config2,
                                                                  main_dir_path=Path(MAIN_PATH),
                                                                  startind=3000)
sigma_conv_c3 = graphics.get_data_for_sigma_convergence_from_toml(toml_config3,
                                                                  main_dir_path=Path(MAIN_PATH),
                                                                  startind=3000)
sigma_conv_c4 = graphics.get_data_for_sigma_convergence_from_toml(toml_config4,
                                                                  main_dir_path=Path(MAIN_PATH),
                                                                  startind=3000)

# Plot QoI path and uncertainty
g=1.2
fig1 = plt.figure(figsize=(10*g, 5*g))#, constrained_layout=True)
spec = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0.25)

colors = sns.color_palette()

ax0 = plt.subplot(spec[0])
p1, = ax0.plot(qoi_dict_c1['x'], qoi_dict_c1['y'], color=colors[0], label='')
ax0.fill_between(qoi_dict_c1['x'],
                 qoi_dict_c1['y']-qoi_dict_c1['sigma_post'],
                 qoi_dict_c1['y']+qoi_dict_c1['sigma_post'], facecolor=colors[0], alpha=0.5)

p2, = ax0.plot(qoi_dict_c2['x'], qoi_dict_c2['y'], color=colors[1], label='')
ax0.fill_between(qoi_dict_c2['x'],
                 qoi_dict_c2['y']-qoi_dict_c2['sigma_post'],
                 qoi_dict_c2['y']+qoi_dict_c2['sigma_post'], facecolor=colors[1], alpha=0.5)

p3, = ax0.plot(qoi_dict_c3['x'], qoi_dict_c3['y'], color=colors[2], label='')
ax0.fill_between(qoi_dict_c3['x'],
                 qoi_dict_c3['y']-qoi_dict_c3['sigma_post'],
                 qoi_dict_c3['y']+qoi_dict_c3['sigma_post'], facecolor=colors[2], alpha=0.5)

p4, = ax0.plot(qoi_dict_c4['x'], qoi_dict_c4['y'], color=colors[3], label='')
ax0.fill_between(qoi_dict_c4['x'],
                 qoi_dict_c4['y']-qoi_dict_c4['sigma_post'],
                 qoi_dict_c4['y']+qoi_dict_c4['sigma_post'], facecolor=colors[3], alpha=0.5)

ax0.set_xlabel('Time (yrs)')
ax0.set_ylabel(r'$QoI: VAF$ $(m^3)$')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper right')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])

p1_prior, = ax1.semilogy(qoi_dict_c1['x'], qoi_dict_c1['sigma_prior'],
             color=colors[0], linestyle='dashed', label='')
p1_post, = ax1.semilogy(qoi_dict_c1['x'], qoi_dict_c1['sigma_post'],
             color=colors[0], label='')

p2_prior, = ax1.semilogy(qoi_dict_c2['x'], qoi_dict_c2['sigma_prior'],
             color=colors[1], linestyle='dashed', label='')
p2_post, = ax1.semilogy(qoi_dict_c2['x'], qoi_dict_c2['sigma_post'],
             color=colors[1], label='')

p3_prior, = ax1.semilogy(qoi_dict_c3['x'], qoi_dict_c3['sigma_prior'],
             color=colors[2], linestyle='dashed', label='')
p3_post, = ax1.semilogy(qoi_dict_c3['x'], qoi_dict_c3['sigma_post'],
             color=colors[2], label='')

p4_prior, = ax1.semilogy(qoi_dict_c4['x'], qoi_dict_c4['sigma_prior'],
             color=colors[3], linestyle='dashed', label='')
p4_post, = ax1.semilogy(qoi_dict_c4['x'], qoi_dict_c4['sigma_post'],
             color=colors[3], label='')

ax1.grid(True, which="major", ls="-")
ax1.set_xlabel('Time (yrs)')
ax1.set_ylabel(r'$\sigma$ $(m^3)$')

at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper right')
ax1.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'results_Qoi_path_experiments.png'), bbox_inches='tight', dpi=150)

# Plot Sigma QoI path and uncertainty change vs No. of eigen values!

fig2 = plt.figure(figsize=(5*g, 10*g))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 1, wspace=0.3, hspace=0.25)

colors = sns.color_palette()

ax0 = plt.subplot(spec[0])
ax0.plot(sigma_conv_c1['eignum'], sigma_conv_c1['sig'],
         color=colors[0], label='subsample-training-step-middle-8E+0')
ax0.plot(sigma_conv_c2['eignum'], sigma_conv_c2['sig'],
         color=colors[1], label='std-adjusted-subsample-training-step-middle-8E aka golden toml')
ax0.plot(sigma_conv_c3['eignum'], sigma_conv_c3['sig'],
         color=colors[2], label='std-adjusted-complete')
ax0.plot(sigma_conv_c4['eignum'], sigma_conv_c4['sig'],
         color=colors[3], label='std-adjusted-subsample-training-step-middle-4E')
ax0.grid()
ax0.legend(frameon=True, bbox_to_anchor=(1.1, 1), ncol=1, fontsize=18)

ax0.set_xlabel('No. of Eigen values')
ax0.set_ylabel(r'$\sigma$ $(m^3)$')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper right')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.semilogy(sigma_conv_c1['ind'],
             np.abs(np.diff(sigma_conv_c1['sig']))/np.diff(sigma_conv_c1['eignum']),
             color=colors[0])
ax1.plot(sigma_conv_c1['ind2'],
         np.exp(sigma_conv_c1['slope'] * sigma_conv_c1['ind2'] + sigma_conv_c1['inter']),
         color=colors[0], alpha=0.5)
ax1.text(11000, 1e10, 'slope=' + str(round(sigma_conv_c1['slope'], 5)) +
         '\n R^2=' + str(round(sigma_conv_c1['result'].rvalue**2, 3)),
         fontsize=18, color=colors[0])

ax1.semilogy(sigma_conv_c2['ind'],
             np.abs(np.diff(sigma_conv_c2['sig']))/np.diff(sigma_conv_c2['eignum']),
             color=colors[1])
ax1.plot(sigma_conv_c2['ind2'],
         np.exp(sigma_conv_c2['slope'] * sigma_conv_c2['ind2'] + sigma_conv_c2['inter']),
         color=colors[1], alpha=0.5)
ax1.text(11000, 1e9, 'slope=' + str(round(sigma_conv_c2['slope'], 5)) +
         '\n R^2=' + str(round(sigma_conv_c2['result'].rvalue**2, 3)),
         fontsize=18, color=colors[1])

ax1.semilogy(sigma_conv_c3['ind'],
             np.abs(np.diff(sigma_conv_c3['sig']))/np.diff(sigma_conv_c3['eignum']),
             color=colors[2])
ax1.plot(sigma_conv_c3['ind2'],
         np.exp(sigma_conv_c3['slope'] * sigma_conv_c3['ind2'] + sigma_conv_c3['inter']),
         color=colors[2], alpha=0.5)
ax1.text(11000, 1e8, 'slope=' + str(round(sigma_conv_c3['slope'], 5)) +
         '\n R^2=' + str(round(sigma_conv_c3['result'].rvalue**2, 3)),
         fontsize=18, color=colors[2])

ax1.semilogy(sigma_conv_c4['ind'],
             np.abs(np.diff(sigma_conv_c4['sig']))/np.diff(sigma_conv_c4['eignum']),
             color=colors[3])
ax1.plot(sigma_conv_c4['ind2'],
         np.exp(sigma_conv_c4['slope'] * sigma_conv_c4['ind2'] + sigma_conv_c4['inter']),
         color=colors[3], alpha=0.5)
ax1.text(11000, 1e7, 'slope=' + str(round(sigma_conv_c4['slope'], 5)) +
         '\n R^2=' + str(round(sigma_conv_c4['result'].rvalue**2, 3)),
         fontsize=18, color=colors[3])

ax1.grid()

ax1.set_xlabel('No. of Eigen values')
ax1.set_ylabel(r'$\delta$$\sigma$ $(m^3)$')

at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper right')
ax1.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'results_deltaQoi_path_experiments.png'), bbox_inches='tight', dpi=150)
