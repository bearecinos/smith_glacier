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

import os

import sys
from pathlib import Path
from fenics_ice import model, solver, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
import pickle
import numpy as np


def run_fwds(config_file_itslive, config_file_measures):
    """Run the forward part of the simulation with a
    modified alpha and beta and save ONLY THE QOI"""

    # Read configuration files from measures
    params_m = ConfigParser(config_file_measures)

    # Load the static model data (geometry, smb, etc)
    input_data_m = inout.InputData(params_m)

    # Get model mesh, This does not really matter as the mesh is the same!
    mesh_m = fice_mesh.get_mesh(params_m)

    # Define the model object for MEaSUREs
    mdl_m = model.model(mesh_m, input_data_m, params_m)

    mdl_m.alpha_from_inversion()
    mdl_m.beta_from_inversion()

    # Read run config file from ITSLIVE
    params_i = ConfigParser(config_file_itslive)
    inout.log_preamble("forward", params_i)

    # We need this for the output
    outdir_i = params_i.io.output_dir
    phase_name_i = params_i.time.phase_name

    # Load the static model data (geometry, smb, etc)
    # Input data should be the same as measures but just in case
    # I define new objects with ITSLive toml
    input_data_i = inout.InputData(params_i)

    # Define the model object for ITSLIVE
    mdl_i = model.model(mesh_m, input_data_i, params_i)

    mdl_i.alpha_from_inversion()
    mdl_i.beta_from_inversion()

    # TODO: help here is needed, can I replace mdl_i.alpha via this?:
    mdl_i.alpha.assign(mdl_i.alpha + (mdl_m.alpha - mdl_i.alpha)*1/100)

    mdl_i.beta.assign(mdl_i.beta + (mdl_m.beta - mdl_i.beta)*1/100)

    # Solve
    slvr = solver.ssa_solver(mdl_i, mixed_space=params_i.inversion.dual)
    slvr.save_ts_zero()

    qoi_func = slvr.get_qoi_func()

    # Run the forward model
    Q = slvr.timestep(adjoint_flag=1, qoi_func=qoi_func)

    # Now we save the new QoI as pickle
    # Lets gather the data and the file output directory
    run_length = params_i.time.run_length
    num_sens = params_i.time.num_sens
    t_sens = np.flip(np.linspace(run_length, 0, num_sens))

    out_dir = Path(outdir_i)/phase_name_i

    qoi_file = os.path.join(str(out_dir),
                            "_".join((params_i.io.run_name,
                                      params_i.time.phase_suffix +
                                      "qoi_only.p")))

    with open(qoi_file, 'wb') as pfile:
        pickle.dump([Q, t_sens], pfile)

    return Q


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Expected two configuration files (*.toml)"
    run_fwds(sys.argv[1], sys.argv[2])
