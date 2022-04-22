"""
Useful functions to organise data, compute vertex for fenic_ice paramters,
 and other utils that did not fit in any other module, lets try to keep
 this one short!
"""

import logging
import numpy as np
import os
import math
import fnmatch
import re
import pandas as pd
from decimal import Decimal
from functools import reduce
import operator
from .backend import MPI, Mesh, XDMFFile, Function, \
    project, sqrt, HDF5File, Measure, TestFunction, TrialFunction, assemble, inner
from dolfin import KrylovSolver
from tlm_adjoint.interface import function_new

# Module logger
log = logging.getLogger(__name__)

def get_netcdf_extent(netcdf_dataset):
    """Get extent of netcdf dataset"""
    extent = {}
    extent['xmin'] = netcdf_dataset.variables['x'][0]
    extent['xmax'] = netcdf_dataset.variables['x'][-1]
    extent['ymin'] = netcdf_dataset.variables['y'][0]
    extent['ymax'] = netcdf_dataset.variables['y'][-1]

    assert extent['xmin'] < extent['xmax']
    assert extent['ymin'] < extent['ymax']
    return extent

def slice_by_xy(arr, xx, yy, extent):
    """Return the section of an array (and xx, yy) which are within bounds"""

    assert arr.shape == (yy.shape[0], xx.shape[0])
    x_inds = np.where((xx >= extent["xmin"]) & (xx <= extent["xmax"]))[0]
    y_inds = np.where((yy >= extent["ymin"]) & (yy <= extent["ymax"]))[0]

    # surely there's a better way to use np.where here?
    sliced_arr = arr[y_inds[0]:y_inds[-1]+1, x_inds[0]:x_inds[-1]+1]

    return sliced_arr, xx[x_inds], yy[y_inds]

def paterson(ttemp):
    """
    paterson and budd 1982 (??)
    takes temperature in Celsius (not Kelvin)
    TODO: Check units with Dan
    """

    ttrip = 273.15
    ttc = 263.15

    aa1 = 1.14e-5 / 3600.0 / 24.0 / 365.0
    qq1 = 60.0e3
    aa2 = 5.471e10 / 3600.0 / 24.0 / 365.0

    qq2 = 139.0e3
    rr = 8.314

    ttempk = ttrip + ttemp

    ff = 1.0e-9

    I1 = ttempk < ttc
    I2 = ttempk >= ttc

    at = np.zeros(np.shape(ttempk))

    at[I1] = aa1 * np.exp(-qq1 / rr / ttempk[I1])
    at[I2] = aa2 * np.exp(-qq2 / rr / ttempk[I2])

    return at

def getTomlItem(dictionary, key_list):
    return reduce(operator.getitem, key_list, dictionary)

def setTomlItem(dictionary, key_list, value):
    param = reduce(operator.getitem, key_list[:-1], dictionary)
    param[key_list[-1]] = value

def composeName(root, suff, value):
    assert isinstance(value, float)
    value_str = "%1.0e" % value
    return "_".join((root, suff, value_str))

def composeFileName(suff, value):
    assert isinstance(value, float)
    value_str = "%.E" % value
    mantissa, exp = value_str.split('E')
    new = mantissa + 'E' + exp[0] + exp[2:]
    new_name = suff + '_' + new
    return new_name

def get_path_for_experiment(path, experiment_name):
    """
    Read .csv files for each l-curve experiment and append
    all results in a single pandas data frame

    :param path to experiment file
    :param experiment name e.g. gamma_alpha, gamma_beta
    :return: path strings with the inv J_cost function results
    for each l-curve experiment.
    """
    j_paths = []
    dir_path = os.path.join(path, experiment_name)

    for root, dirs, files in os.walk(dir_path):
        dirs.sort()
        files = [os.path.join(root, f) for f in files]
        excludes = ['*inversion_progress*', '*.xml', '*.h5', '*.xdmf']
        excludes = r'|'.join([fnmatch.translate(x) for x in excludes]) or r'$.'
        j_paths = [f for f in files if not re.match(excludes, f)]

    return j_paths[0]

def get_data_for_experiment(j_paths):
    """
    Read .csv files for each l-curve experiment and append
    all results in a single pandas data frame

    :param path to experiment files
    :return: pandas.Dataframe with the inv J_cost function results
    for each l-curve experiment.
    """
    ds = pd.DataFrame()
    for file in j_paths:
        ds = pd.concat([ds, pd.read_csv(file)], axis=0)

    # reset the index
    ds.reset_index(drop=True, inplace=True)

    return ds

def get_xml_from_exp(path=str, experiment_name=str, var_name=str, var_value=str):
    """
    Finds the path of an xml file with the parameter field result
    from the inversion, estimated with a specific value
    in the l-curve experiment (e.g. field estimated with a
    min gamma_alpha)
    :param path to xml file
    :param experiment name (e.g. gamma_alpha, gamma_beta)
    :return: xml file path
    """
    xml_f = []
    dir_path = os.path.join(path, experiment_name)
    for root, dirs, files in os.walk(dir_path):
        dirs.sort()
        files = [os.path.join(root, f) for f in files]
        includes = ['*' + var_value + '_' + var_name + '.xml']
        includes = r'|'.join([fnmatch.translate(x) for x in includes]) or r'$.'
        xml_f = [f for f in files if re.match(includes, f)]
    return xml_f[0]

def compute_vertex_for_parameter_field(xml_file, param_space, dg_space, mesh_in):
    """
    Compute vertex values for a specific parameter

    xml_file: path to the parameter xml file
    param_space: either FunctionSpace(mesh_in, 'Lagrange',3)
    dg_space: a FunctionSpace(mesh_in, 'DG', 0)
    mesh_in: mesh to plot to
    :return: vertex values for that parameter
    """
    # Build the space functions
    parameter = Function(param_space, xml_file)
    # Project each to the mesh
    param_proj = project(parameter, dg_space)

    # Return vertex values for each parameter function in the mesh
    return param_proj.compute_vertex_values(mesh_in)



def compute_vertex_for_velocity_field(xml_file, v_space, q_space, mesh_in):
    """
    Compute vertex values for a specific parameter

    xml_file: path to the velocity output xml file
    v_space: a VectorFunctionSpace(mesh_in,'Lagrange', 1, dim=2) or
            a fice_mesh.get_periodic_space(params, mesh_in, dim=2)
    q_space: a FunctionSpace(mesh_in, 'Lagrange',1)
    mesh_in: mesh to plot to
    :return: vertex values for that parameter
    """
    # Build the space functions
    vel = Function(v_space, xml_file)
    u, v = vel.split()
    uv = project(sqrt(u * u + v * v), q_space)

    # Return vertex values for each parameter function in the mesh
    return uv.compute_vertex_values(mesh_in)

def compute_vertex_for_dV_components(dV,
                                     mesh,
                                     hd5_fpath=str,
                                     var_name=str,
                                     n_item=int,
                                     mult_mmatrix=False):
    """
    Compute vertex values for a specific parameter and
    split them into the different components of the dual
    space (alpha and beta)

    :param Q: Function Space
    :param dV: Function
    :param mesh: finite element mesh already in fenics ice format
    :param hd5_fpath: path to .h5 file to read
    :param var_name: name of the variable to read
    :param n_item: number of item either n_sens or number of eigenvectors
    :param mult_mmatrix: do we want to multiply by the mass matrix
    default False, so No.
    :return: va, vb vertex values of each vector component
    """

    hdf5data = HDF5File(MPI.comm_world, hd5_fpath, 'r')

    name_field = var_name + '/vector_'

    hdf5data.read(dV, f'{name_field}{n_item}')

    dx = Measure('dx', domain=mesh)
    Q = dV.function_space()
    Qp_test, Qp_trial = TestFunction(Q), TrialFunction(Q)

    # Mass matrix solver
    M_mat = assemble(inner(Qp_trial, Qp_test) * dx)


    M_solver = KrylovSolver(M_mat.copy(), "cg", "sor")  # DOLFIN KrylovSolver object
    M_solver.parameters.update({"relative_tolerance": 1.0e-14,
                                "absolute_tolerance": 1.0e-32})


    this_action = function_new(dV, name=f"M_inv_action")

    M_solver.solve(this_action.vector(), dV.vector())

    if mult_mmatrix:
        dV_alpha, dV_beta = this_action.split(deepcopy=True)
    else:
        dV_alpha, dV_beta = dV.split(deepcopy=True)

    # Vector to plot
    va = dV_alpha.compute_vertex_values(mesh)
    vb = dV_beta.compute_vertex_values(mesh)

    return va, vb


def normalise_data_array(array):
    """

    :param array: numpy array to normalize
    :return: array values between -1 and 1
    """
    first_n = (array - np.amin(array)) / (np.amax(array) - np.amin(array))
    first_norm = 2 * first_n - 1
    return first_norm

def standarise_data_array(array):
    """

    :param array: numpy array to standarised
    :return: array standarised
    """
    A = (array - np.mean(array)) / np.std(array)
    return A

def centre_data_array(array):
    """

    :param array: numpy array to standarised
    :return: array standarised
    """
    A = array - np.mean(array)
    return A

def generate_parameter_configuration_range(cov_m,
                                           len_m,
                                           save_path,
                                           target_param,
                                           length_constant=False):
    """
    :param cov_m: middle point to generate a convariance range
    :param length: middle point to generate a lenght scale range
    :param length_constant: if True we keep this contant
    :return: gamma, delta
    """
    fac = 3
    btm = [len_m * (fac ** i) for i in range(3, -1, -1)]
    top = [len_m / (fac ** i) for i in range(4)]
    len_a = np.unique(np.concatenate((btm, top)))
    p = np.sqrt(1 / (4 * math.pi * cov_m))
    np.savetxt(os.path.join(save_path, target_param+'_length_scale_range.txt'),
               len_a, delimiter=',', fmt='%f')
    gamma = len_a * p
    delta = (1 / len_a) * p

    # We fix the covariance and vary the length scale
    if length_constant:
        # We fix the length scale and vary the covariance
        btm = [cov_m * (fac ** i) for i in range(3, -1, -1)]
        top = [cov_m / (fac ** i) for i in range(4)]
        cov_a = np.unique(np.concatenate((btm, top)))
        print('Covariance will vary from', cov_a)
        print('Length scale will stay constant ', len_m)
        p = np.sqrt(1 / (4 * math.pi * cov_a))
        gamma = len_m * p
        delta = (1 / len_m) * p
        np.savetxt(os.path.join(save_path, target_param +'_cov_range.txt'),
                   cov_a, delimiter=',', fmt='%f')

    return gamma, delta

def generate_constant_parameter_configuration(target_param=str,
                                              cov_c=None,
                                              len_c=None):
    """

    :param target_param: alpha of beta
    :param cov_m: constant covariance for the parameter
    :param len_m: constant length scale for the parameter
    :return: gamma and delta for the parameter that will remained constant
    """

    p = np.sqrt(1 / (4 * math.pi * cov_c))
    gamma = cov_c * p
    delta = (1 / len_c) * p

    print(f'gamma_{target_param} will be fix to ', "{:.1E}".format(Decimal(gamma)))
    print(f'delta_{target_param} will be fix to ', "{:.1E}".format(Decimal(delta)))

    return gamma, delta

