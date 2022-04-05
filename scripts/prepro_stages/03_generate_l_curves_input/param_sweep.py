"""
Tool for generating TOMLs for parameter sweeps in fenics_ice

Input:
------
-conf: repository config.ini file path
-name_toml_template: name of the .toml template (fenics_ice configuration file)
-target_param: the parameter to be swept (e.g. 'delta_beta')
-name_sweep: acronym to identify this sweep in run_lcurves_inv etc
           e.g. 'db' for delta_beta
param_min, param_max: start and end of sweep values (inclusive!)
command_template: this helps generate a bash script which will run each
           simulation in turn.

By default this sweeps orders of magnitude but this can be controlled by
editing the call to np.geomspace.

@authors: Fenics_ice contributors
"""
import sys
import os
import toml
import numpy as np
from pathlib import Path
import argparse
from configobj import ConfigObj
from decimal import Decimal
import re

# Argument passer
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str,
                    default="../../../config.ini", help="pass config file")
parser.add_argument("-name_toml_template", type=str,
                    default="", help="pass toml template path")
parser.add_argument("-target_param", type=str,
                    default="delta_beta",
                    help="the parameter to be swept e.g. delta_beta")
parser.add_argument("-name_sweep", type=str,
                    default="db",
                    help="acronym to identify this sweep in run names"
                         " etc e.g. db for delta_beta")
parser.add_argument("-min", type=np.float64,
                    default=1.0e2,
                    help="minimum value for start sweep (inclusive)")
parser.add_argument("-max", type=np.float64,
                    default=1.0e10,
                    help="max value for start sweep (inclusive)")

args = parser.parse_args()

# Get smith_glacier repo configuration file
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Main directory path
MAIN_PATH = config['main_path']
sys.path.append(MAIN_PATH)

from ficetools import utils_funcs

# Get .toml file template
template_full_path = os.path.join(os.environ['PREPRO_STAGES'],
                                  '03_generate_l_curves_input/'+ args.name_toml_template)
toml_name = Path(template_full_path)
assert toml_name.exists(), f".toml file template {toml_name} not found"

# target_param = ["inversion", "delta_beta"]
target_param = args.target_param
name_suff = args.name_sweep

# Set parameter range
param_min = args.min
param_max = args.max

runs_directory = os.path.join(MAIN_PATH, 'scripts/run_experiments/run_lcurves')
tomls_f = os.path.join(runs_directory, 'tomls_'+target_param)
output_dir = os.path.join(MAIN_PATH, 'output/04_run_inv_lcurves/output')
diag_dir = os.path.join(MAIN_PATH, 'output/04_run_inv_lcurves/diagnostics')

# Paths to data
if not os.path.exists(tomls_f):
    os.makedirs(tomls_f)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(diag_dir):
    os.makedirs(diag_dir)

command_template = "mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py " \
                   "$RUN_CONFIG_DIR/run_lcurves/{target_param}/{toml_fname} |& tee {log_fname}\n"

script_name = f"sweep_{target_param}.sh"

name_param = ["inversion", "phase_suffix"]
# Regexes
phase_suffix = re.compile(rf"^(phase_suffix = ).*")
param_re = re.compile(rf"^({target_param} = ).*")

# Compute round steps if possible
steps = int(np.log10(param_max) - np.log10(param_min)) + 1

# toml writes "np.float*" types as strings, so convert to list of floats here
param_range = np.geomspace(param_min, param_max, steps, dtype=np.float64).tolist()
print('will sweep over', param_range)

template = toml.load(toml_name)
template_name = utils_funcs.getTomlItem(template, name_param)
script_file = open(os.path.join(tomls_f,script_name), 'w')

for i in range(steps):

    phase_suffix_name = utils_funcs.composeName(template_name, name_suff, param_range[i])
    filename = Path(tomls_f, utils_funcs.composeName(toml_name.stem,
                                                     name_suff,
                                                     param_range[i])).with_suffix(".toml")

    with open(toml_name, 'r') as inny:
        lines = inny.readlines()

        with open(filename, 'w') as outy:
            for line in lines:
                if phase_suffix.match(line):
                    new_line = phase_suffix.match(line).group(1) + '\''f"{target_param}_" + "{:.1E}".format(Decimal(param_range[i]))+ '\'' + "\n"
                    print(new_line)
                elif param_re.match(line):
                    #new_line = param_re.match(line).group(1) + "%e\n" % param_range[i]
                    new_line = param_re.match(line).group(1) + "{:.1E}".format(Decimal(param_range[i])) + "\n"
                    print(new_line)
                else:
                    new_line = line
                outy.write(new_line)

    # append to shell script
    cmd = command_template.replace("{toml_fname}", filename.name).\
        replace("{log_fname}", f"log_{phase_suffix_name}").\
        replace("{target_param}", f"tomls_{target_param}")
    script_file.write(cmd)

