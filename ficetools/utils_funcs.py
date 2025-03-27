"""
Useful functions to organise data, compute vertex for fenic_ice paramters,
 and other utils that did not fit in any other module, lets try to keep
 this one short!
"""

import logging
import numpy as np
import os
from pathlib import Path
import math
import fnmatch
import re
import pandas as pd
import pickle
from collections import defaultdict
from decimal import Decimal
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from pyproj import Transformer
import xarray as xr
from functools import reduce
import operator
from .backend import MPI, Mesh, XDMFFile, Function, FunctionSpace, \
    project, sqrt, HDF5File, Measure, TestFunction, TrialFunction, assemble, inner
from dolfin import KrylovSolver
from tlm_adjoint.interface import function_new
import h5py
from fenics_ice import config, mesh, inout, model, solver
from ufl import finiteelement

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

def normalize(array, percentile=None):
    """
    :param array: array to normalise
    :param percentile: percentile to normalise
                        if none normalises to maximum
    :return: normedarray
    """
    if percentile is not None:
        absmax = percentile
    else:
        absmax = np.max(np.abs(array))

    normedarray = array / absmax

    return normedarray


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
    if not length_constant:
        print('Length scale will vary from', len_a)
        print('Covariance will stay constant ', cov_m)

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

def define_stage_output_paths(params, stage, diagnostics=False):
    """
    Defines the stages output dirs for a particular set of params

    :param params: params read from the toml
    :param stage: stage of the model workflow that we need
    :return: exp_outdir: string with the path for the stage requested
    """

    # general paths
    if diagnostics:
        out_dir = params.io.diagnostics_dir
    else:
        out_dir = params.io.output_dir

    # general names of stages
    phase_name_inversion = params.inversion.phase_name
    phase_name_fwd = params.time.phase_name
    phase_name_eigen = params.eigendec.phase_name
    phase_name_errp = params.error_prop.phase_name
    phase_name_invsigma = params.inv_sigma.phase_name

    # stages suffixes
    phase_suffix_inversion = params.inversion.phase_suffix
    phase_suffix_fwd = params.time.phase_suffix
    phase_suffix_eigen = params.eigendec.phase_suffix
    phase_suffix_errprop = params.error_prop.phase_suffix
    phase_suffix_invsigma = params.inv_sigma.phase_suffix

    exp_outdir = None

    if stage == 'inversion':
        exp_outdir = Path(out_dir) / phase_name_inversion / phase_suffix_inversion
    if stage == 'time':
        exp_outdir = Path(out_dir) / phase_name_fwd / phase_suffix_fwd
    if stage == 'eigendec':
        exp_outdir = Path(out_dir) / phase_name_eigen / phase_suffix_eigen
    if stage == 'error_prop':
        exp_outdir = Path(out_dir) / phase_name_errp / phase_suffix_errprop
    if stage == 'inv_sigma':
        exp_outdir = Path(out_dir) / phase_name_invsigma / phase_suffix_invsigma

    return exp_outdir

def get_file_names_for_path_plot(params):
    """
    This constructs the file names for the variables
    needed in the QoI sigma path plot

    :param params: params read from the toml
    :return: Q_filename: the 'Qval_ts.p' file name with the corresponding suffix
    :return: sigma_filename: the 'sigma.p' file name with the corresponding suffix
    :return: sigma_prior_file: the 'sigma_prior.p' file name with the corresponding suffix
    """

    # stages suffixes
    phase_suffix_fwd = params.time.phase_suffix
    phase_suffix_errprop = params.error_prop.phase_suffix

    # File names
    Qfile_name = "_".join((params.io.run_name + phase_suffix_fwd, 'Qval_ts.p'))
    sigma_file = "_".join((params.io.run_name + phase_suffix_errprop, 'sigma.p'))
    sigma_prior_file = "_".join((params.io.run_name + phase_suffix_errprop, 'sigma_prior.p'))

    return Qfile_name, sigma_file, sigma_prior_file


def get_file_names_for_inversion_plot(params):
    """
    This constructs the file names for the variables
    needed in the QoI sigma path plot

    :param params: params read from the toml
    :return: dict_inv_fnames: directory with all the file names
        needed for the inversion plot and their corresponding file suffix

    """

    # File names
    alpha_fname = "_".join((params.io.run_name + params.inversion.phase_suffix,
                            '_alpha.xml'))
    beta_fname = "_".join((params.io.run_name + params.inversion.phase_suffix,
                           '_beta.xml'))
    model_vel_fname = "_".join((params.io.run_name + params.inversion.phase_suffix,
                                '_U.xml'))
    obs_vel_fname = "_".join((params.io.run_name + params.inversion.phase_suffix,
                              '_uv_cloud.xml'))

    dict_inv_fnames = {'alpha_fname': alpha_fname,
                       'beta_fname': beta_fname,
                       'model_vel_fname': model_vel_fname,
                       'obs_vel_fname': obs_vel_fname}

    return dict_inv_fnames


def get_file_names_for_invsigma_plot(params):
    """
    This constructs the file names for the variables
    needed in the STD sigma alpha / beta plot

    :param params:  params read from the toml
    :return: file_salpha: sigma alpha file name
             file_sbeta: sigma beta file name
    """

    # stages suffixes
    phase_suffix_inv_sigma = params.inv_sigma.phase_suffix

    # File names
    file_salpha = "_".join((params.io.run_name + phase_suffix_inv_sigma, 'sigma_alpha.xml'))
    file_sbeta = "_".join((params.io.run_name + phase_suffix_inv_sigma, 'sigma_beta.xml'))

    return file_salpha, file_sbeta


def get_pts_from_h5_velfile(file):
    """
    Counts how many data points exist in the velocity files
    :param file: .h5 velocity file
    :return: length of x-coordinates, length of y-coordinates
    """
    f = h5py.File(file, 'r')
    x = f['x_cloud'][:]
    y = f['y_cloud'][:]
    return len(x), len(y)

def get_prior_information_from_toml(toml):
    """
    Gets the prior strength information from a toml file path
    toml: path to toml
    returns: pandas.Dataframe df containing all the run information
    """
    f_name = os.path.basename(toml)
    name, ext = os.path.splitext(f_name)
    list_vars = name.split('_')
    C0a2 = []
    L0a = []
    C0b2 = []
    L0b = []
    vel_config = []

    for var in list_vars:
        if 'C0a2' in var:
            C0a2 = np.append(C0a2, np.float64(var.split('-')[-1]))
        if 'L0a' in var:
            L0a = np.append(L0a, np.float64(var.split('-')[-1]))
        if 'C0b2' in var:
            C0b2 = np.append(C0b2, np.float64(var.split('-')[-1]))
        if 'L0b' in var:
            L0b = np.append(L0b, np.float64(var.split('-')[-1]))
        if 'itslive' in var:
            vel_config = np.append(vel_config, var)

    params = config.ConfigParser(toml)

    dict_prior = {'c0a2': C0a2[0],
                  'c0a': C0a2[0] ** 2,
                  'L0a': L0a[0],
                  'gamma_alpha': params.inversion.gamma_alpha,
                  'delta_alpha': params.inversion.delta_alpha,
                  'c0b2': C0b2[0],
                  'c0b': C0b2[0] ** 2,
                  'L0b': L0b[0],
                  'gamma_beta': params.inversion.gamma_beta,
                  'delta_beta': params.inversion.delta_beta,
                  'vel_file_config': vel_config[0],
                  'path_to_toml': toml}

    df = pd.DataFrame(dict_prior.items())
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    return df

def compute_vertex_for_dQ_dalpha_component(params, n_sen=int, mult_mmatrix=False):
    """
    Compute sensitivities for VAF trajectories
    :params configuration parser from fenics ice
    returns
    --------
    va: dq/da^2
    """
    # Read common data and get mesh information
    mesh_in = mesh.get_mesh(params)

    el = finiteelement.FiniteElement("Lagrange", mesh_in.ufl_cell(), 1)
    mixedElem = el * el

    Q = FunctionSpace(mesh_in, mixedElem)
    dQ = Function(Q)

    # Reading the sensitivity output
    outdir = params.io.output_dir
    phase_name_fwd = params.time.phase_name
    run_name = params.io.run_name
    phase_suffix = params.time.phase_suffix

    fwd_outdir = Path(outdir) / phase_name_fwd / phase_suffix
    file_qts = "_".join((run_name + phase_suffix, 'dQ_ts.h5'))
    hdffile = fwd_outdir / file_qts
    assert hdffile.is_file(), "File not found"

    hdf5data = HDF5File(MPI.comm_world, str(hdffile), 'r')
    name_field = 'dQdalphaXbeta' + '/vector_'
    hdf5data.read(dQ, f'{name_field}{n_sen}')

    # Reading inversion output
    input_data = inout.InputData(params)
    # Define the model
    mdl = model.model(mesh_in, input_data, params)
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()

    # Solve
    slvr = solver.ssa_solver(mdl, mixed_space=params.inversion.dual)
    cntrl = slvr.get_control()

    dQ.vector()[:] = dQ.vector()[:] / (2.0 * cntrl[0].vector()[:])
    dQ.vector().apply("insert")

    dx = Measure('dx', domain=mesh_in)

    Q = dQ.function_space()
    Qp_test, Qp_trial = TestFunction(Q), TrialFunction(Q)

    # Mass matrix solver
    M_mat = assemble(inner(Qp_trial, Qp_test) * dx)

    M_solver = KrylovSolver(M_mat.copy(), "cg", "sor")  # DOLFIN KrylovSolver object
    M_solver.parameters.update({"relative_tolerance": 1.0e-14,
                                "absolute_tolerance": 1.0e-32})

    this_action = function_new(dQ, name=f"M_inv_action")
    M_solver.solve(this_action.vector(), dQ.vector())

    if mult_mmatrix:
        dQ_alpha, dQ_beta = this_action.split(deepcopy=True)
    else:
        dQ_alpha, dQ_beta = dQ.split(deepcopy=True)


    va = dQ_alpha.compute_vertex_values(mesh_in)
    vb = dQ_beta.compute_vertex_values(mesh_in)

    return va, vb

def find_itslive_file(year, path):
    """
    year: float indicating the year
    path: general path to itslive data
    """
    name = 'ANT_G0240_' + str(year) + '.nc'

    for root, dirs, files in os.walk(path):
        if name in files:
            print(name)
            return os.path.join(root, name)

def find_measures_file(year, path):
    """
    year: float indicating the year for measures this will be the
          year that indicates the end of the observational period
    path: general path to measures data
    """
    year_two = str(year)
    year_one = str(year-1)

    name = 'Antarctica_ice_velocity_' + year_one + '_' + year_two + '_1km_v01.nc'

    for root, dirs, files in os.walk(path):
        if name in files:
            print(name)
            return os.path.join(root, name)

def get_vel_ob_sens_dict(params):
    """
    This function gets the dictionary of the output from the
    vel_obs_sens model task, which calculates QoI sensitivities
    to velocity observations (dQ_du and dQ_dv).

    :params Config params for the specific experiment
    returns
    --------
    out: out python dictionary with the results of the simulation
    """
    ## Reading the data
    phase_name = params.obs_sens.phase_name
    phase_suffix = params.obs_sens.phase_suffix

    outdir = params.io.diagnostics_dir
    path_to_file = Path(outdir) / phase_name / phase_suffix

    # We have to define this as it is not defined in fenics_ice
    file_name = 'vel_sens.pkl'
    full_path = os.path.join(path_to_file, file_name)

    file_vel_orig = Path(params.io.input_dir) / params.obs.vel_file
    assert file_vel_orig.is_file()
    vel_file = inout.read_vel_obs(file_vel_orig, model=None, use_cloud_point=False)

    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            out = pickle.load(f)

        assert np.all(out['u_obs']) == np.all(vel_file['u_obs'])
        assert np.all(out['v_obs']) == np.all(vel_file['v_obs'])
        assert np.all(out['uv_obs_pts']) == np.all(vel_file['uv_obs_pts'])

        assert len(out['u_obs']) == len(vel_file['u_std'])

        out['u_std'] = vel_file['u_std']
        out['v_std'] = vel_file['v_std']
    else:
        out = vel_file.copy()

    return out

def project_dq_dc_to_mesh(array, M, vtx_M, wts_M, mesh_in):
    """
    This function projects each derivative into the
    mesh coordinates and computes the vertex values
    in order to plot them on a map

    :params array, array to interpolate into a function
    :params M function space
    :params vtx_M, x coordinates of the new space in the mesh
    :params wts_M,  y coordinates of the new space in the mesh
    :params mesh_in, model mesh

    returns
    --------
    dq_dc: derivatives as functions projected to the model mesh
    """
    dQ_dC = []

    dQ_dC.append(Function(M, static=True))

    dQ_dC[0].vector()[:] = model.interpolate(array, vtx_M, wts_M)

    dq_dc = dQ_dC[0].compute_vertex_values(mesh_in)
    # dq_dc[dq_dc < 0] = 0.0

    return dq_dc

def convert_vel_sens_output_into_functions(out,
                                           M_coords,
                                           M,
                                           periodic_bc,
                                           mesh_in):
    """
    Converts vel obs sens output into functions by
    interpolating the data back in to the model mesh
    TODO by Dan: this should be done by Dan script during the run.

    :params out, python dictonary containing the output of vel_sens.pkl
    :params M_coords, Model function space coordinates
    :params M function space
    :params vtx_M, x coordinates of the new space in the mesh
    :params wts_M,  y coordinates of the new space in the mesh
    :params mesh_in, model mesh

    returns
    --------
    output_copy: a copy of the vel_sens.pkl directory but with all the
    arrays converted into functions that have been interpolated
    to the model mesh coordinates.
    """

    # only do the M space
    vtx_M, wts_M = model.interp_weights(out['uv_obs_pts'],
                                        M_coords,
                                        periodic_bc)

    ## Process first velocity components from the observations
    U = []
    V = []

    U.append(Function(M, static=True))
    V.append(Function(M, static=True))

    U[0].vector()[:] = model.interpolate(out['u_obs'], vtx_M, wts_M)
    V[0].vector()[:] = model.interpolate(out['v_obs'], vtx_M, wts_M)

    out_copy = out.copy()

    out_copy['u_obs'] = U[0].compute_vertex_values(mesh_in)
    out_copy['v_obs'] = V[0].compute_vertex_values(mesh_in)

    ## Process each derivative for every n_sens
    for n_sen in np.arange(15):
        # get the derivative to process
        out_copy['dObsU'][n_sen] = project_dq_dc_to_mesh(out['dObsU'][n_sen], M, vtx_M, wts_M, mesh_in)
        out_copy['dObsV'][n_sen] = project_dq_dc_to_mesh(out['dObsV'][n_sen], M, vtx_M, wts_M, mesh_in)

    return out_copy

def get_alpha_and_beta_points(params):
    """
    Gets alpha and beta functions for a given experiment and inversion
    :params fenics ice parameter configuration
    Returns
    ------
    alpha and beta fields already vectorised as numpy arrays.
    """
    # Reading mesh
    mesh_in = mesh.get_mesh(params)

    # Compute the function spaces from the Mesh
    Q = FunctionSpace(mesh_in, 'Lagrange', 1)
    M = FunctionSpace(mesh_in, 'DG', 0)

    if not params.mesh.periodic_bc:
        Qp = Q
    else:
        Qp = mesh.get_periodic_space(params, mesh_in, dim=1)

    # Now we read Alpha and Beta fields from each run
    # The diagnostic path is the same for both runs
    path_diag = os.path.join(params.io.diagnostics_dir, 'inversion')

    phase_suffix = params.inversion.phase_suffix
    exp_outdir = Path(path_diag) / phase_suffix
    file_alpha = "_".join((params.io.run_name + phase_suffix, 'alpha.xml'))
    file_bglen = "_".join((params.io.run_name + phase_suffix, 'beta.xml'))

    alpha_fp = exp_outdir / file_alpha
    bglen_fp = exp_outdir / file_bglen

    assert alpha_fp.is_file(), "File not found"
    assert bglen_fp.is_file(), "File not found"

    # Define function spaces for alpha only and uv_comp
    alpha_func = Function(Qp, str(alpha_fp))
    alpha_proj = project(alpha_func, M)
    alpha = alpha_proj.compute_vertex_values(mesh_in)

    # Beta space
    beta_func = Function(Qp, str(bglen_fp))
    beta_proj = project(beta_func, M)
    beta = beta_proj.compute_vertex_values(mesh_in)

    return alpha, beta

def dot_product_per_parameter_pair(params_me, params_il):
    """
    Get forward results of QoI sensitivities to params
    for a given experiment.
    :params fenics ice parameter configuration for measures
    :params fenics ice parameter configuration for itslive
    Returns
    ------

    """
    # Read mesh
    mesh_in = mesh.get_mesh(params_me)

    # Read output data to plot
    # Same output dir for both runs
    out_il = params_il.io.output_dir
    phase_name = params_il.time.phase_name
    run_name_il = params_il.io.run_name

    out_me = params_me.io.output_dir
    run_name_me = params_me.io.run_name

    phase_suffix_il = params_il.time.phase_suffix
    phase_suffix_me = params_me.time.phase_suffix

    fwd_outdir_il = Path(out_il) / phase_name / phase_suffix_il
    fwd_outdir_me = Path(out_me) / phase_name / phase_suffix_me

    file_qts_il = "_".join((params_il.io.run_name + phase_suffix_il, 'dQ_ts.h5'))
    file_qts_me = "_".join((params_me.io.run_name + phase_suffix_me, 'dQ_ts.h5'))

    hdffile_il = fwd_outdir_il / file_qts_il
    hdffile_me = fwd_outdir_me / file_qts_me

    assert hdffile_il.is_file(), "File not found"
    assert hdffile_me.is_file(), "File not found"

    el = finiteelement.FiniteElement("Lagrange", mesh_in.ufl_cell(), 1)
    mixedElem = el * el

    Q_f = FunctionSpace(mesh_in, mixedElem)
    dQ_f = Function(Q_f)

    # Get the n_sens
    num_sens = np.arange(0, params_il.time.num_sens)
    print('Years')
    print(num_sens)

    # Get sensitivity output for dQ/da and dQ/db
    # along each num_sen
    fwd_v_alpha_il = defaultdict(list)
    fwd_v_beta_il = defaultdict(list)
    fwd_v_alpha_me = defaultdict(list)
    fwd_v_beta_me = defaultdict(list)

    results_dot_alpha = []
    results_dot_beta = []

    # So we know to which num_sen the sensitivity
    # output belongs
    nametosum = 'fwd_n_'

    for n in num_sens:
        # We get the sensitivities for every num_sen
        # as a vector component
        # For measures
        dq_dalpha_me, dq_dbeta_me = compute_vertex_for_dV_components(dQ_f,
                                                                     mesh_in,
                                                                     str(hdffile_me),
                                                                     'dQdalphaXbeta',
                                                                     n,
                                                                     mult_mmatrix=False)
        # We do not need to multiply by the mass matrix!
        # more info check the compute_vertex_for_dV_component()
        dq_dalpha_il, dq_dbeta_il = compute_vertex_for_dV_components(dQ_f,
                                                                     mesh_in,
                                                                     str(hdffile_il),
                                                                     'dQdalphaXbeta',
                                                                     n,
                                                                     mult_mmatrix=False)

        alpha_v_me, beta_v_me = get_alpha_and_beta_points(params_me)
        alpha_v_il, beta_v_il = get_alpha_and_beta_points(params_il)

        dot_alpha_me = np.dot(dq_dalpha_me, alpha_v_me - alpha_v_il)
        dot_beta_me = np.dot(dq_dbeta_me, beta_v_me - beta_v_il)


        #fwd_v_alpha_me[f'{nametosum}{n}'].append(dq_dalpha_me)
        #fwd_v_beta_me[f'{nametosum}{n}'].append(dq_dbeta_me)

        #fwd_v_alpha_il[f'{nametosum}{n}'].append(dq_dalpha_il)
        #fwd_v_beta_il[f'{nametosum}{n}'].append(dq_dbeta_il)

        results_dot_alpha.append(dot_alpha_me)
        results_dot_beta.append(dot_beta_me)

    return results_dot_alpha, results_dot_beta


def contour_to_lines(contour):
    """
    Turn a sigle contour lines into LineString
    contour: single contour form a contour matplotlib plt.
    """
    lines = []
    for collection in contour.collections:
        for path in collection.get_paths():
            v = path.vertices
            line = LineString(v)
            lines.append(line)
    return lines


def re_grid_model_output(x, y, output, resolution=240, mask_xarray=xr.DataArray, return_points=False):
    """
    x: easting coordinates
    y: north coordinate
    output: model output to re-grid
    resolution: new grid resolution e.g. 240 m same as ITSLIVE grid
    mask: a xr.Array mask over the ocean

    Note that this does not account for a mask in the ocean, thus will
    interpolate values there as well.
    """

    xi = np.arange(x.min(), x.max(), resolution)
    yi = np.arange(y.min(), y.max(), resolution)

    grid_x, grid_y = np.meshgrid(xi, yi)

    output_gridded = griddata((x, y), output, (grid_x, grid_y), method='linear')

    xm, ym = np.meshgrid(mask_xarray.x.data, mask_xarray.y.data)

    # Flatten original coordinates and mask values
    points = np.column_stack((xm.ravel(), ym.ravel()))
    values = mask_xarray.data.ravel()

    mask_bm_2d = griddata(points, values, (grid_x, grid_y), method='nearest')

    # We mask the ocean from bedmachine mask
    masked_output = np.where(mask_bm_2d != 0, output_gridded, np.nan)

    if return_points:
        return masked_output, grid_x, grid_y
    else:
        return masked_output


def model_grounding_line_to_shapefile(params,
                                      time_step=240):
    """
    Convert model grounding line to shapefile
    params: Fenics_ice params object
    time_step: year of gronding line to output
    """

    # Reading the model output
    outdir = params.io.diagnostics_dir
    phase_name_fwd = params.time.phase_name
    phase_suffix = params.time.phase_suffix

    fwd_outdir_me = Path(outdir) / phase_name_fwd / phase_suffix

    file_name = 'H_timestep_' + str(time_step) + '.xml'

    file_H_time = "_".join((params.io.run_name + phase_suffix, file_name))
    xmlfile_H = fwd_outdir_me / file_H_time

    assert xmlfile_H.is_file(), "File not found"

    # Compute the function spaces from the Mesh
    mesh_in = mesh.get_mesh(params)
    input_data = inout.InputData(params)
    mdl = model.model(mesh_in, input_data, params)
    M = mdl.M

    H = Function(M, str(xmlfile_H))
    H_vertex = H.compute_vertex_values(mesh_in)

    # Reading the inversion output
    phase_name_inversion = params.inversion.phase_name
    inv_outdir_me = Path(outdir) / phase_name_inversion / phase_suffix
    bed_file = "_".join((params.io.run_name + phase_suffix, 'bed.xml'))

    bed_file_fullpath = inv_outdir_me / bed_file
    assert bed_file_fullpath.is_file(), "File not found"

    Q = mdl.Q
    bed_func = Function(Q, str(bed_file_fullpath))
    bed_v = bed_func.compute_vertex_values(mesh_in)

    rhoi = 917.0  # Density of ice
    rhow = 1030.0  # Density of sea water

    H_AF = H_vertex + bed_v * (rhow / rhoi)
    mask = H_AF > 0.0  # Example value to filter
    masked_H_AF = np.ma.array(H_AF, mask=mask)

    interm_masked_H_AF = np.where((masked_H_AF.mask == 0) | (masked_H_AF.mask == 1),
                                  masked_H_AF.mask, 0)

    x = mesh_in.coordinates()[:, 0]
    y = mesh_in.coordinates()[:, 1]

    xi = np.arange(x.min(), x.max(), 240)
    yi = np.arange(y.min(), y.max(), 240)
    grid_x, grid_y = np.meshgrid(xi, yi)

    masked_H_AF_2d = griddata((x, y), interm_masked_H_AF,
                              (grid_x, grid_y), method='linear')

    plt.ioff()
    contour_1 = plt.contour(grid_x, grid_y,
                            masked_H_AF_2d, levels=[0, 1], colors='blue')
    plt.clf()

    # Extract line geometries from each contour
    lines = contour_to_lines(contour_1)

    # Optionally assign attributes (e.g., source)
    geoms = lines
    attrs = ['H'+ str(time_step)] * len(lines)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'source': attrs, 'geometry': geoms})

    # Set the correct CRS (same as your grid_x, grid_y — likely EPSG:3031)
    gdf.set_crs(epsg=3031, inplace=True)

    return gdf


def interpolate_line(line: LineString, step: float = 10.0):
    """
    Generates a list of evenly spaced points along a LineString geometry,
    at a fixed interval (e.g., every 1,000 meters).
    line: LineString (transect from a shapefile geometry)
    step: how to divide the transect e.g. 10 meters
    """
    n_points = int(line.length / step)
    distances = np.linspace(0, line.length, n_points)
    return [line.interpolate(d) for d in distances]


def interp_model_output_to_centreline(ds, gdf, n_sens, step):
    """
    ds: xarray.DataArray with the model output
    gdf: geopandas data frame containing the centreline or
    transect
    n_sens: array of years to interpolate
    step: how to divide the centreline e.g. 10 meters
    """

    transect = gdf.geometry.iloc[0]

    # here we do every 1km along the main centreline
    points = interpolate_line(transect, step=step)

    lons = np.array([pt.x for pt in points])
    lats = np.array([pt.y for pt in points])

    dQ_dM_y0 = ds['dQ_dM_' + str(n_sens[0])].interp(x=("points", lons), y=("points", lats))
    dQ_dM_y1 = ds['dQ_dM_' + str(n_sens[-1])].interp(x=("points", lons), y=("points", lats))

    # We return the points close to the ocean first
    return dQ_dM_y0, dQ_dM_y1


def interp_model_mask_to_centreline(ds_y0, ds_y1, gdf, years, step):
    """
    ds: xarray.DataArray with the model output
    gdf: geopandas data frame containing the centreline or
    transect
    years: years of the simulation
    step: how to divide the centreline e.g. 10 meters
    """

    transect = gdf.geometry.iloc[0]

    # here we do every 1km along the main centreline
    points = interpolate_line(transect, step=step)

    lons = np.array([pt.x for pt in points])
    lats = np.array([pt.y for pt in points])

    mask_y0 = ds_y0['mask_' + str(years[0])].interp(x=("points", lons), y=("points", lats))
    mask_y1 = ds_y1['mask_' + str(years[-1])].interp(x=("points", lons), y=("points", lats))

    # We return the points close to the ocean first
    return mask_y0, mask_y1







