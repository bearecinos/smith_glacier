"""
Plot run_invsigma output, Paths of QIS along T's in smith exp.

- Reads output data (stored in .h5)
- Plots things in a multiplot grid

@authors: Fenics_ice contributors
"""
import pickle
import seaborn as sns
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from configobj import ConfigObj
import argparse
from fenics_ice import config

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str,
                    default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

#Load main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Load configuration file for more order in paths
configuration = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

run_files = os.path.join(MAIN_PATH, 'scripts/run_experiments/run_workflow')
toml = os.path.join(run_files, 'smith.toml')

output_dir = os.path.join(MAIN_PATH, 'output/03_run_inv')

params = config.ConfigParser(toml, top_dir=Path(MAIN_PATH))

Qfile = os.path.join(output_dir, params.io.run_name + '_Qval_ts.p')
sigmafile = os.path.join(output_dir, params.io.run_name + '_sigma.p')
sigmapriorfile = os.path.join(output_dir, params.io.run_name + '_sigma_prior.p')

with open(Qfile, 'rb') as f:
    out = pickle.load(f)
dQ_vals = out[0]
dQ_t = out[1]

with open(sigmafile, 'rb') as f:
    out = pickle.load(f)
sigma_vals = out[0]
sigma_t = out[1]

with open(sigmapriorfile, 'rb') as f:
    out = pickle.load(f)
sigma_prior_vals = out[0]

sigma_interp = np.interp(dQ_t, sigma_t, sigma_vals)
sigma_prior_interp = np.interp(dQ_t, sigma_t, sigma_prior_vals)

x = dQ_t
#y = dQ_vals - dQ_vals[0]
y2 = dQ_vals[0] - dQ_vals
s = 2*sigma_interp
sp = 2*sigma_prior_interp

from matplotlib import rcParams
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

g=1.2
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec

fig1 = plt.figure(figsize=(10*g, 5*g))#, constrained_layout=True)
spec = gridspec.GridSpec(1, 2, wspace=0.25, hspace=0.25)

colors = sns.color_palette()

ax0 = plt.subplot(spec[0])
ax0.plot(x, y2, color='k')
#ax0.fill_between(x, y2-sp, y2+sp, facecolor=colors[0], alpha=0.5, label='sp')
ax0.fill_between(x, y2-s, y2+s, facecolor=colors[1], alpha=0.5)
ax0.set_xlabel('Time (yrs)')
ax0.set_ylabel(r'$Q$ $(m^4)$')
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.semilogy(x, sp, label='prior projection')
ax1.semilogy(x, s, label='sigma projection')
ax1.legend()
ax1.set_xlabel('Time (yrs)')
ax1.set_ylabel(r'$\sigma$ $(m^4)$')
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc='upper left')
ax1.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'sigma_paths_vaf.png'), bbox_inches='tight')