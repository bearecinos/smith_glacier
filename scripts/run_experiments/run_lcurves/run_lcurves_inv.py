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

from fenics_ice.backend import HDF5File, project

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path

from fenics_ice import model, solver, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
import datetime

def run_inv(config_file):
    """Run the inversion part of the simulation"""
    # Read run config file
    params = ConfigParser(config_file)

    inout.setup_logging(params)
    inout.log_preamble("inverse", params)

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    # Get the model mesh
    mesh = fice_mesh.get_mesh(params)
    mdl = model.model(mesh, input_data, params)

    # pts_lengthscale = params.obs.pts_len

    mdl.gen_alpha()

    # Add random noise to Beta field iff we're inverting for it
    mdl.bglen_from_data()
    mdl.init_beta(mdl.bglen_to_beta(mdl.bglen), pert=False)

    # Next line will output the initial guess for alpha fed into the inversion
    # File(os.path.join(outdir,'alpha_initguess.pvd')) << mdl.alpha

    #####################
    # Run the Inversion #
    #####################

    slvr = solver.ssa_solver(mdl)
    slvr.inversion()

    ##############################################
    #  Write out variables in outdir and         #
    #  diagnostics folder                        #
    #############################################

    phase_name = params.inversion.phase_name
    phase_suffix = params.inversion.phase_suffix
    outdir = Path(params.io.output_dir) / phase_name / phase_suffix
    diag_dir = Path(params.io.diagnostics_dir)

    # Required for next phase (HDF5):

    invout_file = params.io.inversion_file

    phase_suffix = params.inversion.phase_suffix
    if len(phase_suffix) > 0:
        invout_file = params.io.run_name + phase_suffix + '_invout.h5'

    invout = HDF5File(mesh.mpi_comm(), str(outdir/invout_file), 'w')

    invout.parameters.add("gamma_alpha", slvr.gamma_alpha)
    invout.parameters.add("delta_alpha", slvr.delta_alpha)
    invout.parameters.add("gamma_beta", slvr.gamma_beta)
    invout.parameters.add("delta_beta", slvr.delta_beta)
    invout.parameters.add("delta_beta_gnd", slvr.delta_beta_gnd)
    invout.parameters.add("timestamp", str(datetime.datetime.now()))

    invout.write(mdl.alpha, 'alpha')
    invout.write(mdl.beta, 'beta')

    # For visualisation (XML & VTK):
    if params.io.write_diagnostics:

        inout.write_variable(slvr.U, params, outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)
        inout.write_variable(mdl.beta,
                             params, outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)

        inout.write_variable(mdl.alpha, params, outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)

        Bglen = project(slvr.beta_to_bglen(slvr.beta), mdl.M)
        Bglen.rename("Bglen", "")
        inout.write_variable(Bglen, params, outdir=diag_dir,
                             phase_name=phase_name, phase_suffix=phase_suffix)

    return mdl


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_inv(sys.argv[1])

