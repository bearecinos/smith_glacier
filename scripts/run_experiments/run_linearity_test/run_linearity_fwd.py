# For fenics_ice copyright information see ACKNOWLEDGEMENTS in the fenics_ice
# root directory

# This file is part of fenics_ice.
#
# fenics_ice is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# fenics_ice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.
from fenics_ice.backend import *
from fenics_ice.backend import HDF5File, project

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import pandas as pd
from fenics_ice import model, solver, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
from IPython import embed

import datetime

def run_fwds(config_file_itslive, config_file_measures):
    """Run the forward part of the simulation with a
    modified alpha and beta"""

    # Read configuration files from measures
    params_m = ConfigParser(config_file_measures)

    # Load the static model data (geometry, smb, etc)
    input_data_m = inout.InputData(params_m)

    # Get model mesh
    mesh_m = fice_mesh.get_mesh(params_m)

    # Define the model
    mdl_m = model.model(mesh_m, input_data_m, params_m)

    mdl_m.alpha_from_inversion()
    mdl_m.beta_from_inversion()

    # Read run config file
    params_i = ConfigParser(config_file_itslive)
    log_i = inout.setup_logging(params_i)
    inout.log_preamble("forward", params_i)

    outdir_i = params_i.io.output_dir
    diag_dir_i = params_i.io.diagnostics_dir
    phase_name_i = params_i.time.phase_name

    # Load the static model data (geometry, smb, etc)
    input_data_i = inout.InputData(params_i)

    # Get model mesh
    mesh_i = fice_mesh.get_mesh(params_i)

    # Define the model
    mdl_i = model.model(mesh_i, input_data_i, params_i)

    mdl_i.alpha_from_inversion()
    mdl_i.beta_from_inversion()

    mdl_i.alpha = mdl_i.alpha + (mdl_m.alpha - mdl_i.alpha)*1/100
    function_update_state(mdl_i.alpha)

    mdl_i.beta = mdl_i.beta + (mdl_m.beta - mdl_i.beta)*1/100
    function_update_state(mdl_i.beta)

    # Solve
    slvr = solver.ssa_solver(mdl_i, mixed_space=params_i.inversion.dual)
    slvr.save_ts_zero()

    cntrl = slvr.get_control()

    qoi_func = slvr.get_qoi_func()

    # TODO here - cntrl now returns a list - so compute_gradient returns a list of tuples

    # Run the forward model
    Q = slvr.timestep(adjoint_flag=1, qoi_func=qoi_func)
    # Run the adjoint model, computing gradient of Qoi w.r.t cntrl
    dQ_ts = compute_gradient(Q, cntrl)  # Isaac 27

    # Output model variables in ParaView+Fenics friendly format
    # Output QOI & DQOI (needed for next steps)
    inout.write_qval(slvr.Qval_ts, params_i)
    inout.write_dqval(dQ_ts, [var.name() for var in cntrl], params_i)

    # Output final velocity, surface & thickness (visualisation)
    inout.write_variable(slvr.U, params_i, name="U_fwd",
                         outdir=diag_dir_i, phase_name=phase_name_i,
                         phase_suffix=params_i.time.phase_suffix)
    inout.write_variable(mdl_i.surf, params_i, name="surf_fwd",
                         outdir=diag_dir_i, phase_name=phase_name_i,
                         phase_suffix=params_i.time.phase_suffix)

    H = project(mdl_i.H, mdl_i.Q)
    inout.write_variable(H, params_i, name="H_fwd",
                         outdir=diag_dir_i, phase_name=phase_name_i,
                         phase_suffix=params_i.time.phase_suffix)
    return mdl_i



if __name__ == "__main__":
    assert len(sys.argv) == 3, "Expected two configuration files (*.toml)"
    run_fwds(sys.argv[1], sys.argv[2])

