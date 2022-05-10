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
import logging
import sys
import os
import toml
import numpy as np
from pathlib import Path
import argparse
from configobj import ConfigObj
from decimal import Decimal
import re
import math

# Argument passer
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str,
                    default="../../../config.ini", help="pass config file")
parser.add_argument("-name_toml_template", type=str,
                    default="", help="pass toml template path")
parser.add_argument("-target_param", type=str,
                    default="beta",
                    help="the parameter to be swept e.g. beta")
parser.add_argument("-name_sweep", type=str,
                    default="bc",
                    help="acronym to identify this sweep in run names"
                         " etc e.g. bc for gamma_beta when covariance "
                         "vary")
parser.add_argument("-cov", type=np.float64,
                    default=100.0,
                    help="middle value for covariance sweep range (inclusive)")
parser.add_argument("-len", type=np.float64,
                    default=100.0,
                    help="middle value for the length scale sweep range (inclusive)")
parser.add_argument("-len_constant",
                    action="store_true",
                    help="If true we will only vary the covariance between gamma "
                         "(e.g. regularization param: gamma_beta) "
                         "with its own change (e.g. delta_beta). "
                         "If false we will vary the length scale instead")

args = parser.parse_args()

# Get smith_glacier repo configuration file
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Main directory path
MAIN_PATH = config['main_path']
sys.path.append(MAIN_PATH)

from ficetools.utils_funcs import generate_constant_parameter_configuration, \
    generate_parameter_configuration_range, getTomlItem, composeName

# Get .toml file template
template_full_path = os.path.join(os.environ['PREPRO_STAGES'],
                                  '03_generate_l_curves_input/'+ args.name_toml_template)
toml_name = Path(template_full_path)
assert toml_name.exists(), f".toml file template {toml_name} not found"

# target_param = ["inversion", "delta_beta"]
target_param = args.target_param
name_suff = args.name_sweep

cov_m = args.cov
len_m = args.len

# Set the middle point for the different ranges
runs_directory = os.path.join(MAIN_PATH, 'scripts/run_experiments/run_lcurves')
tomls_f = os.path.join(runs_directory, 'tomls_'+target_param+'_'+name_suff)
output_dir = os.path.join(MAIN_PATH, 'output/04_run_inv_lcurves_cov/output')
diag_dir = os.path.join(MAIN_PATH, 'output/04_run_inv_lcurves_cov/diagnostics')

# Paths to data
if not os.path.exists(tomls_f):
    os.makedirs(tomls_f)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(diag_dir):
    os.makedirs(diag_dir)

command_template = "mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py " \
                   "$RUN_CONFIG_DIR/run_lcurves/{target_param}_{name_suff}/{toml_fname} |& tee {log_fname}\n"

script_name = f"sweep_{target_param}_{name_suff}.sh"

name_param = ["inversion", "phase_suffix"]
# Regexes
phase_suffix = re.compile(rf"^(phase_suffix = ).*")
param_re_gamma = re.compile(rf"^(gamma_{target_param} = ).*")
param_re_delta = re.compile(rf"^(delta_{target_param} = ).*")
param_re_delta_beta_gnd = re.compile(rf"^(delta_beta_gnd = ).*")


range_gamma, range_delta = generate_parameter_configuration_range(cov_m,
                                                                  len_m,
                                                                  target_param=target_param,
                                                                  save_path=runs_directory,
                                                                  length_constant=args.len_constant)

print(f'gamma_{target_param} will be vary from', range_gamma)
print(f'delta_{target_param} will be vary from', range_delta)

if target_param == 'beta':
    print('delta_beta_gnd will be vary from', range_delta)

template = toml.load(toml_name)
template_name = getTomlItem(template, name_param)
script_file = open(os.path.join(tomls_f,script_name), 'w')

steps = range(len(range_gamma))

for i in steps:
    print(range_gamma[i])
    phase_suffix_name = composeName(template_name, name_suff, range_gamma[i])
    filename = Path(tomls_f, composeName(toml_name.stem,
                                         name_suff,
                                         range_gamma[i])).with_suffix(".toml")
    print(filename)

    with open(toml_name, 'r') as inny:
        lines = inny.readlines()

        with open(filename, 'w') as outy:
            for line in lines:
                if phase_suffix.match(line):
                    new_line = phase_suffix.match(line).group(1) + '\''f"{target_param}_{name_suff}_" + "{:.0E}".format(Decimal(range_gamma[i]))+ '\'' + "\n"
                    print(new_line)
                elif param_re_gamma.match(line):
                    new_line = param_re_gamma.match(line).group(1) + "{:.2E}".format(Decimal(range_gamma[i])) + "\n"
                    print(new_line)
                elif param_re_delta.match(line):
                    new_line = param_re_delta.match(line).group(1) + "{:.2E}".format(Decimal(range_delta[i])) + "\n"
                    print(new_line)
                elif target_param == 'beta' and param_re_delta_beta_gnd.match(line):
                    new_line = param_re_delta_beta_gnd.match(line).group(1) + "{:.2E}".format(Decimal(range_delta[i])) + "\n"
                    print(new_line)
                else:
                    new_line = line
                outy.write(new_line)

    #append to shell script
    cmd = command_template.replace("{toml_fname}", filename.name).\
        replace("{name_suff}", name_suff).\
        replace("{log_fname}", f"log{phase_suffix_name}").\
        replace("{target_param}", f"tomls_{target_param}")
    script_file.write(cmd)

