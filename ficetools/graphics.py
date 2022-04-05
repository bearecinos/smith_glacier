"""
A set of tools for plotting input data and all the experiments results
from fenics_ice
"""
import numpy as np
import logging

#Plotting imports
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
from matplotlib.offsetbox import AnchoredText

# Module logger
log = logging.getLogger(__name__)

def plot_field_in_contour_plot(x, y, t, field, field_name,
                               ax, vmin=None, vmax=None, cmap=None, add_mesh=False):
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

    # Get mesh stuff
    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]
    t = mesh.cells()
    trim = tri.Triangulation(x, y, t)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    c = ax.tricontourf(x, y, t, data_array, levels=levels, cmap=cmap)
    #ax.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
    cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
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