"""
A set of tools for plotting input data and all the experiments results
from fenics_ice
"""
import numpy as np
import logging
import salem
import pyproj

#Plotting imports
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
from matplotlib.offsetbox import AnchoredText

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
    c = ax.tricontourf(x, y, t, field, levels=levels, cmap=cmap)
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
                        rot=float):
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
    for i, lab in enumerate(data_frame[var_name]):
        ax.annotate(lab, (div[i], j_ls[i]), xytext=xytext, textcoords='offset pixels',
                     horizontalalignment='right',
                     verticalalignment='bottom', rotation=rot, size=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel('J_ls')
    ax.set_xlabel('J_reg/' + str(var_name))
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