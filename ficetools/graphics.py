"""
A set of tools for plotting input data and all the experiments results
from fenics_ice
"""
import numpy as np
import logging
import salem
import pyproj
import pickle

#Plotting imports
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
from matplotlib.offsetbox import AnchoredText
from scipy.stats import linregress

from fenics_ice import model, config
from fenics_ice import mesh as fice_mesh
from fenics import *
from pathlib import Path

# Module logger
log = logging.getLogger(__name__)

def plot_field_in_contour_plot(x, y, t, field, field_name,
                               ax, vmin=None, vmax=None, levels=None, ticks=None,
                               cmap=None, add_mesh=False):
    """
    Makes a matplotlib tri contour plot of any parameter field
    in a specific axis.

    :param x mesh x coordinates
    :param y mesh y coordinates
    :param field to plot (e.g. alpha, U, etc.)
    :param field_name: name of the variable to plot
    :param ax to plot things
    :param vmin minimum value for the color scale
    :param vmax maximum value for the color scale
    :param cmap: color map
    :param add_mesh: add mesh to contour plot
    :return: {} plot in a specific axis
    """

    trim = tri.Triangulation(x, y, t)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    minv = vmin
    maxv = vmax
    if levels is None:
        levels = np.linspace(minv, maxv, 200)
        ticks = np.linspace(minv, maxv, 3)
    c = ax.tricontourf(x, y, t, field, levels=levels, cmap=cmap, extend="both")
    if add_mesh:
        ax.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
    cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
    cbar.ax.set_xlabel(field_name)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    return {}

def plot_lcurve_scatter(data_frame, var_name, ax,
                        xlim_min=None, xlim_max=None,
                        ylim_min=None, ylim_max=None, xytext=(float, float),
                        rot=float, add_annotate=False):
    """

    :param data_frame: pandas.Dataframe with the results of the l-curve
    :param var_name: parameter name varied in the l-curve
    :param ax: matplotlib.Axes where we will plot  things
    :param xlim_min: lower limit of the x-axis
    :param xlim_max: upper limit of the x-axis
    :param ylim_min: lower limit of the y-axis
    :param ylim_max: upper limit of the y-axis
    :param xytext: how much padding we want on the labels (+x, +y)
    :param rot: rotation of the labels
    :return: scatter plot of the l-curve exp.
    """

    j_ls = data_frame['J_ls'].values
    div = data_frame['J_reg'].values / data_frame[var_name].values

    ax.scatter(div, j_ls)
    ax.plot(div, j_ls)
    if add_annotate:
        for i, lab in enumerate(data_frame[var_name]):
            ax.annotate(lab, (div[i], j_ls[i]), xytext=xytext, textcoords='offset pixels',
                         horizontalalignment='right',
                         verticalalignment='bottom', rotation=rot, size=12)
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_ylabel(r'$J_{c}$')
    if 'alpha' in var_name:
        ax.set_xlabel(r'$J_{reg}/$' + r'$\gamma_{\alpha}$')
    if 'beta' in var_name:
        ax.set_xlabel(r'$J_{reg}/$' + r'$\gamma_{\beta}$')
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    return {}

def plot_field_in_tricontourf(data_array,
                              mesh,
                              ax,
                              varname=str,
                              num_eigen=int,
                              ticks=None,
                              levels=None,
                              add_text=True):
    """

    :param data_array: field to plot (e.g. eigenvector)
    :param varname: field component to plot (e.g. eigenvector alpha)
    :param num_eigen: vector number to plot
    :param mesh: mesh to take out coordinates information
    :param ax: axes number to plot the data
    :param ticks: ticks on colorbar
    :param levels: levels
    :return: {} plot
    """
    import seaborn as sns
    cmap = sns.color_palette("RdBu", as_cmap=True)

        # Get mesh triangulation
    x, y, t = read_fenics_ice_mesh(mesh)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    c = ax.tricontourf(x, y, t, data_array, levels=levels, cmap=cmap)
    #ax.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
    cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal", extend="both")
    cbar.ax.set_xlabel('dE '+f'{varname}')
    if add_text:
        n_text = AnchoredText('eigenvector = ' + str(num_eigen),
                              prop=dict(size=12), frameon=True, loc='upper right')
        ax.add_artist(n_text)

    return {}

def make_colorbar_scale(array, percnt):
    """

    :param array:
    :return: levels and ticks
    """
    fifty = np.quantile(array, percnt)
    level = np.linspace(fifty, fifty*-1, 100)
    ticks = [min(level), np.median(level), max(level)]
    return level, ticks

def read_fenics_ice_mesh(mesh_in, retur_trim=False):
    """

    :param mesh_in: Fenics_ice mesh output
    :return: x, y, and t
    """
    x = mesh_in.coordinates()[:, 0]
    y = mesh_in.coordinates()[:, 1]
    t = mesh_in.cells()

    if retur_trim:
        trim = tri.Triangulation(x, y, t)
        return x, y, t, trim
    return x, y, t

def define_salem_grid(dataset):
    """
    Define a Salem grid according to a velocity xarray
    grid
    :param dataset: xarray data set
    :return: salem.Grid as g
    """

    proj = pyproj.Proj('EPSG:3413')

    y = dataset.y
    x = dataset.x

    dy = abs(y[0] - y[1])
    dx = abs(x[0] - x[1])

    # Pixel corner
    origin_y = y[0] + dy * 0.5
    origin_x = x[0] - dx * 0.5

    g = salem.Grid(nxny=(len(x), len(y)), dxdy=(dx, -1*dy),
                   x0y0=(origin_x, origin_y), proj=proj)

    return g

def get_projection_grid_labels(smap):
    """
    Print stereographic projection lables
    :param smap:
    :return: lon_lables, lat_lables
    """

    # Change XY into interval coordinates, and back after rounding
    xx, yy = smap.grid.pixcorner_ll_coordinates
    _xx = xx / 1.0
    _yy = yy / 0.5

    mm_x = [np.ceil(np.min(_xx)), np.floor(np.max(_xx))]
    mm_y = [np.ceil(np.min(_yy)), np.floor(np.max(_yy))]

    smap.xtick_levs = (mm_x[0] + np.arange(mm_x[1] - mm_x[0] + 1)) * \
                      1.0
    smap.ytick_levs = (mm_y[0] + np.arange(mm_y[1] - mm_y[0] + 1)) * \
                      0.5

    lon_lables = smap.xtick_levs
    lat_lables = smap.ytick_levs

    # The labels (quite ugly)
    smap.xtick_pos = []
    smap.xtick_val = []
    smap.ytick_pos = []
    smap.ytick_val = []

    _xx = xx[0 if smap.origin == 'lower' else -1, :]
    _xi = np.arange(smap.grid.nx + 1)

    for xl in lon_lables:
        if (xl > _xx[-1]) or (xl < _xx[0]):
            continue
        smap.xtick_pos.append(np.interp(xl, _xx, _xi))
        label = ('{:.' + '1' + 'f}').format(xl)
        label += 'W' if (xl < 0) else 'E'
        if xl == 0:
            label = '0'
        smap.xtick_val.append(label)

    _yy = np.sort(yy[:, 0])
    _yi = np.arange(smap.grid.ny + 1)

    if smap.origin == 'upper':
        _yi = _yi[::-1]

    for yl in lat_lables:
        #((yl > _yy[-1]) or (yl < _yy[0]))
        if (yl > _yy[-1]) or (yl < _yy[0]):
            continue
        smap.ytick_pos.append(np.interp(yl, _yy, _yi))
        label = ('{:.' + '1' + 'f}').format(yl)
        label += 'S' if (yl > 0) else 'N'
        if yl == 0:
            label = 'Eq.'
        smap.ytick_val.append(label)

    return smap.xtick_pos, smap.xtick_val, smap.ytick_pos, smap.ytick_val

def set_levels_ticks_for_colorbar(vmin, vmax):

    levels = np.linspace(vmin, vmax, 200)
    ticks = np.linspace(vmin, vmax, 3)

    return levels, ticks


def define_stage_output_paths(params, stage, diagnostics=False):
    """
    Defines the stages output dirs for a particular set of params

    params: params read from the toml
    stage: stage of the model workflow that we need

    returns
    -------
    dict_output: string with the path for the stage requested
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
    This constructs the file names for the files
    needed for the QoI sigma path plot

    params: params read from the toml

    returns
    -------
    Q_filename: the 'Qval_ts.p' file name with the corresponding suffix
    sigma_filename: the 'sigma.p' file name with the corresponding suffix
    sigma_prior_file: the 'sigma_prior.p' file name with the corresponding suffix
    """

    # stages suffixes
    phase_suffix_fwd = params.time.phase_suffix
    phase_suffix_errprop = params.error_prop.phase_suffix

    # File names
    Qfile_name = "_".join((params.io.run_name + phase_suffix_fwd, 'Qval_ts.p'))
    sigma_file = "_".join((params.io.run_name + phase_suffix_errprop, 'sigma.p'))
    sigma_prior_file = "_".join((params.io.run_name + phase_suffix_errprop, 'sigma_prior.p'))

    return Qfile_name, sigma_file, sigma_prior_file


def get_file_names_for_invsigma_plot(params):
    """
    Return the file names for inversion sigma output data

    :param params:
    :return: file_salpha: sigma alpha file name with phase suffix
             file_sbeta: sigma beta file name with phase suffix
             file_prior_alpha: prior alpha file name with phase suffix
            file_prior_beta: prior beta file name with phase suffix
    """

    # stages suffixes
    phase_suffix_inv_sigma = params.inv_sigma.phase_suffix

    # File names
    file_salpha = "_".join((params.io.run_name + phase_suffix_inv_sigma, 'sigma_alpha.xml'))
    file_sbeta = "_".join((params.io.run_name + phase_suffix_inv_sigma, 'sigma_beta.xml'))

    file_prior_alpha = "_".join((params.io.run_name + phase_suffix_inv_sigma, 'sigma_prior_alpha.xml'))
    file_prior_beta = "_".join((params.io.run_name + phase_suffix_inv_sigma, 'sigma_prior_beta.xml'))

    return file_salpha, file_sbeta, file_prior_alpha, file_prior_beta

def get_data_for_sigma_path_from_toml(toml, main_dir_path):
    """
    Returns the data to plot the QoI uncertainty path with a given .toml

    :param toml: fenics_ice configuration file
    :param main_dir_path: main directory path e.g. /scratch/local/smith_glacier
    :return: qoi_dict: a dictionary with all the data needed for the plot.
            x: number of years
            y: QoI trayectory
            s_c: QoI posterior distribution
            sp_c: QoI prior distribution
    """

    params = config.ConfigParser(toml, top_dir=Path(main_dir_path))

    log.info("The configuration pass is the following:")
    dict_config = {'phase_suffix': params.inversion.phase_suffix,
                    'gamma_alpha': params.inversion.gamma_alpha,
                    'delta_alpha': params.inversion.delta_alpha,
                    'gama_beta': params.inversion.gamma_beta,
                    'delta_beta': params.inversion.delta_beta}
    log.info("-----------------------------------------------")
    log.info(dict_config)

    exp_outdir_fwd = define_stage_output_paths(params, 'time')
    exp_outdir_errp = define_stage_output_paths(params, 'error_prop')

    c_fnames = get_file_names_for_path_plot(params)

    Q_c_fname = c_fnames[0]
    sigma_c_fname = c_fnames[1]
    sigma_c_prior_fname = c_fnames[2]

    # config three
    Qfile_c = exp_outdir_fwd / Q_c_fname
    sigmafile_c = exp_outdir_errp / sigma_c_fname
    sigmapriorfile_c = exp_outdir_errp / sigma_c_prior_fname

    # Check if the file exist
    assert Qfile_c.is_file()
    assert sigmafile_c.is_file()
    assert sigmapriorfile_c.is_file()

    with open(Qfile_c, 'rb') as f:
        out = pickle.load(f)
    dQ_vals_c = out[0]
    dQ_t_c = out[1]

    with open(sigmafile_c, 'rb') as f:
        out = pickle.load(f)
    sigma_vals_c = out[0]
    sigma_t_c = out[1]

    with open(sigmapriorfile_c, 'rb') as f:
        out = pickle.load(f)
    sigma_prior_vals_c = out[0]

    sigma_interp_c = np.interp(dQ_t_c, sigma_t_c, sigma_vals_c)
    sigma_prior_interp_c = np.interp(dQ_t_c, sigma_t_c, sigma_prior_vals_c)

    x_c = dQ_t_c

    y_c = dQ_vals_c - dQ_vals_c[0]

    s_c = 2 * sigma_interp_c
    sp_c = 2 * sigma_prior_interp_c

    qoi_dict = {'x': x_c, 'y': y_c,
                'sigma_post': s_c, 'sigma_prior': sp_c}

    return qoi_dict

def get_params_posterior_std(toml, main_dir_path):
    """
    Returns the data to plot the parameters uncertainty via a given .toml

    :param toml: fenics_ice configuration file
    :param main_dir_path: main directory path e.g. /scratch/local/smith_glacier
    :return: sigma_params_dict: a dictionary with all the data needed for plotting
            pior and posterior distribution of the inverted parameters.
            x: x coordinates
            y: y coordinates
            t: triangulation
            sigma_alpha: posterior alpha std
            sigma_beta: posterior beta std
            prior_alpha: prior alpha std
            prior_alpha: prior beta std
    """
    params = config.ConfigParser(toml, top_dir=Path(main_dir_path))

    exp_outdir_invsigma = define_stage_output_paths(params, 'inv_sigma')

    file_names_invsigma = get_file_names_for_invsigma_plot(params)

    path_alphas = exp_outdir_invsigma / file_names_invsigma[0]
    path_betas = exp_outdir_invsigma / file_names_invsigma[1]

    path_alphas_prior = exp_outdir_invsigma / file_names_invsigma[2]
    path_betas_prior = exp_outdir_invsigma / file_names_invsigma[3]

    assert path_alphas_prior.is_file()
    assert path_betas_prior.is_file()
    assert path_alphas.is_file()
    assert path_betas.is_file()

    # Reading mesh
    mesh_in = fice_mesh.get_mesh(params)

    # Compute the function spaces from the Mesh
    M = FunctionSpace(mesh_in, 'DG', 0)

    x, y, t = read_fenics_ice_mesh(mesh_in)

    alpha_sigma = Function(M, str(path_alphas))
    alpha_sig = project(alpha_sigma, M)
    sigma_alpha = alpha_sig.compute_vertex_values(mesh_in)

    beta_sigma = Function(M, str(path_betas))
    beta_sig = project(beta_sigma, M)
    sigma_beta = beta_sig.compute_vertex_values(mesh_in)

    alpha_sigmap = Function(M, str(path_alphas_prior))
    alpha_sigp= project(alpha_sigmap, M)
    prior_alpha = alpha_sigp.compute_vertex_values(mesh_in)

    beta_sigmap = Function(M, str(path_betas_prior))
    beta_sigp = project(beta_sigmap, M)
    prior_beta = beta_sigp.compute_vertex_values(mesh_in)

    sigma_params_dict = {'x': x, 'y': y, 't': t,
                         'sigma_alpha': sigma_alpha,
                         'sigma_beta': sigma_beta,
                         'prior_alpha': prior_alpha,
                         'prior_beta': prior_beta}

    return sigma_params_dict


def get_data_for_sigma_convergence_from_toml(toml, main_dir_path, startind=3000):
    """
    Returns the data to plot the QoI convergence and uncertainty change
     according to the number of eigen values calculated.
     Per toml!

    :param toml: fenics_ice configuration file
    :param main_dir_path: main directory path e.g. /scratch/local/smith_glacier
    :param startind: which eigen value to start calculating the slop for sigma change
    :return: qoi_conv_dict: a dictionary with all the
            data needed for the convergence plot.
            eigenum: number of eigen values
            sigma: QoI posterior
    """

    params = config.ConfigParser(toml, top_dir=Path(main_dir_path))

    exp_outdir_errp = define_stage_output_paths(params, 'error_prop')
    phase_suffix_errprop = params.error_prop.phase_suffix

    sigma_conv_filename = "".join((params.io.run_name + '_' + phase_suffix_errprop, 'sigma_qoi_convergence.p'))

    sigma_conv_path = exp_outdir_errp / sigma_conv_filename

    # Check if the file exist
    assert sigma_conv_path.is_file()

    with open(sigma_conv_path, 'rb') as f:
        out = pickle.load(f)

    eignum = np.array(out[0])
    sig = out[1]

    ind = 0.5 * (eignum[1:] + eignum[0:-1])
    diffs = (np.diff(sig)) / np.diff(eignum)

    ind2 = ind[ind > startind]
    diffs2 = diffs[ind > startind]

    result = linregress(ind2, np.log(np.abs(diffs2)))
    slope = result.slope
    inter = result.intercept

    sigma_full = np.abs(diffs[-1]) / (1 - np.exp(slope))

    qoi_conv_dict = {'eignum': eignum,
                     'sig': sig,
                     'ind': ind,
                     'ind2': ind2,
                     'result': result,
                     'slope': slope,
                     'inter': inter,
                     'sigma_full': sigma_full}

    return qoi_conv_dict

