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

def run_invs(config_file_train, config_file_test):
    """Run the inversion part of the simulation"""
    # Read run config file
    params_trn = ConfigParser(config_file_train)

    inout.setup_logging(params_trn)
    inout.log_preamble("inverse", params_trn)

    # Load the static model data (geometry, smb, etc)
    input_data_trn = inout.InputData(params_trn)

    # Get the model mesh
    mesh = fice_mesh.get_mesh(params_trn)
    mdl_trn = model.model(mesh, input_data_trn, params_trn)

    mdl_trn.gen_alpha()

    # Add random noise to Beta field iff we're inverting for it
    mdl_trn.bglen_from_data()
    mdl_trn.init_beta(mdl_trn.bglen_to_beta(mdl_trn.bglen), pert=False)

    #####################
    # Run the Inversion #
    #####################

    slvr_trn = solver.ssa_solver(mdl_trn)

    slvr_trn.inversion()

    ##############################################
    #  Write out variables in outdir and         #
    #  diagnostics folder                        #
    #############################################

    phase_name = params_trn.inversion.phase_name
    phase_suffix = params_trn.inversion.phase_suffix
    outdir = Path(params_trn.io.output_dir) / phase_name / phase_suffix
    diag_dir = Path(params_trn.io.diagnostics_dir)

    # Required for next phase (HDF5):

    invout_file = params_trn.io.inversion_file

    phase_suffix = params_trn.inversion.phase_suffix
    if len(phase_suffix) > 0:
        invout_file = params_trn.io.run_name + phase_suffix + '_invout.h5'

    invout = HDF5File(mesh.mpi_comm(), str(outdir/invout_file), 'w')

    invout.parameters.add("gamma_alpha", slvr_trn.gamma_alpha)
    invout.parameters.add("delta_alpha", slvr_trn.delta_alpha)
    invout.parameters.add("gamma_beta", slvr_trn.gamma_beta)
    invout.parameters.add("delta_beta", slvr_trn.delta_beta)
    invout.parameters.add("delta_beta_gnd", slvr_trn.delta_beta_gnd)
    invout.parameters.add("timestamp", str(datetime.datetime.now()))
    invout.write(mdl_trn.alpha, 'alpha')
    invout.write(mdl_trn.beta, 'beta')

    # For visualisation (XML & VTK):
    if params_trn.io.write_diagnostics:
        inout.write_variable(slvr_trn.U, params_trn, outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)

        U_obs = project((mdl_trn.v_cloud_Q ** 2 + mdl_trn.u_cloud_Q ** 2) ** (1.0 / 2.0), mdl_trn.M)
        U_obs.rename("uv_cloud", "")
        inout.write_variable(U_obs, params_trn, name="uv_cloud", outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)

        inout.write_variable(mdl_trn.beta,
                             params_trn, outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)

        inout.write_variable(mdl_trn.alpha, params_trn, outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)

    ##### Now we set up the inversion for the test  ###############

    # Read run config file
    params_test = ConfigParser(config_file_test)

    # Load the static model data (geometry, smb, etc)
    input_data_test = inout.InputData(params_test)

    # Get the model mesh
    mdl_test = model.model(mesh, input_data_test, params_test)

    mdl_test.gen_alpha()

    # Add random noise to Beta field iff we're inverting for it
    mdl_test.bglen_from_data()
    mdl_test.init_beta(mdl_test.bglen_to_beta(mdl_test.bglen), pert=False)


    # Defined our solver object
    slvr_test = solver.ssa_solver(mdl_test)

    # Set reg terms to zero
    slvr_test.zero_inv_params()

    # Set alpha and beta equal to the inversion output from
    # training set
    slvr_test.set_control_fns([mdl_trn.alpha, mdl_trn.beta])

    assert slvr_test.alpha == mdl_trn.alpha
    assert slvr_test.beta == mdl_trn.beta

    J_test = slvr_test.comp_J_inv(verbose=True)

    if params_test.io.write_diagnostics:
        diag_dir = Path(params_test.io.diagnostics_dir)
        phase_suffix = params_test.inversion.phase_suffix
        inout.write_variable(slvr_test.beta,
                             params_test, outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)

        inout.write_variable(slvr_test.alpha, params_test, outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)

    return J_test


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Expected two configuration files (*.toml)"
    run_invs(sys.argv[1], sys.argv[2])

