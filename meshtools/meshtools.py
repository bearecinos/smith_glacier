"""
A set of tools for mesh generation, not particularly well organised.
"""
import os
import numpy as np
import pandas as pd
import h5py
from netCDF4 import Dataset as NCDataset
import rasterio.features
import rasterio.transform
from shapely.geometry import shape, Polygon
import shapely.ops
import dolfin as df
import gmsh
import meshio
import subprocess
from scipy.interpolate import RegularGridInterpolator, griddata
import xarray as xr
import fnmatch
import re
import tempfile
import logging
from fenics import *

#Plotting imports
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri

# Module logger
log = logging.getLogger(__name__)

# TODO - deal with this better
MMG_LIB_LOC = None

def set_mmg_lib(lib_loc):
    global MMG_LIB_LOC
    MMG_LIB_LOC = lib_loc

def get_dx(netcdf_df):
    """Check pixels are square and return dx"""
    xx = netcdf_df.variables['xaxis'][:]
    yy = netcdf_df.variables['yaxis'][:]

    dx = np.unique(xx[1:] - xx[:-1])
    dy = np.unique(yy[1:] - yy[:-1])

    assert len(dx == 1)
    assert len(dy == 1)
    assert dx[0] == dy[0]

    return dx[0]

def get_smith_ice_mask(mask_file, resolve_nunataks=False, x_varname='x', y_varname='y'):
    """Load the ice mask from HDF5"""
    infile = h5py.File(mask_file, 'r')
    mask = infile['data_mask'][:]

    # In the h5 file which is part of the fenics_ice source, the variable is stored (x,y)
    # So transpose it and check
    nx = infile[x_varname].shape[0]
    ny = infile[y_varname].shape[0]

    mask = mask.T

    assert mask.shape == (ny, nx)

    if not resolve_nunataks:
        # Ignore nunataks...
        mask[mask == -10] = 1

        assert(set(np.unique(mask)) == set((0.0, 1.0)))
    else:
        assert(set(np.unique(mask)) == set((0.0, 1.0, -10.0)))

    return mask.astype(np.int32)

def get_affine_transform(infile):
    """
    Given an HDF5 raster file, return the affine transformation
    from row/col to coordinates.
    """

    suff = infile.suffix
    if suff == '.h5':
        inny = h5py.File(infile, 'r')
    elif suff == '.nc':
        inny = NCDataset(infile)
    else:
        raise ValueError(f"Bad filetype: {suff}")

    x = inny['x'][:]
    y = inny['y'][:]

    # Check origin top-left
    assert (y[0] - y[-1] > 0)
    assert (x[0] - x[-1] < 0)

    dy = abs(y[0] - y[1])
    dx = abs(x[0] - x[1])

    # Pixel corner
    origin_y = y[0] + dy*0.5
    origin_x = x[0] - dx*0.5

    # Use rasterio (origin -> transformation)
    affine = rasterio.transform.from_origin(west=origin_x, north=origin_y,
                                            xsize=dx, ysize=dy)

    # Test before returning
    test_transform = rasterio.transform.xy(affine, len(y) - 1, len(x) - 1)
    assert test_transform == (x[-1], y[-1]), 'Bad affine transformation'

    return affine

def get_netcdf_vel_years(netcdf_dataset):
    """Return list of integer years available from 'vx2012' strings"""
    fields = netcdf_dataset.variables.keys()
    years = [int(f[2:]) for f in fields if 'vx' in f]
    years.sort()
    return years

def get_eigenstrain_rate(vx, vy, dx):
    """
    Return the eigenvalues of the strain rate given velocity rasters

    eigenvalues are sorted ascending -  eig[0] < eig[1]
    """
    # vx = netcdf_dataset.variables[f'vx{year}'][:]
    # vy = netcdf_dataset.variables[f'vy{year}'][:]

    # Compute the velocity gradients
    # NB: y axis comes first in these rasters, hence 0,1 -> y,x
    dvx_dy, dvx_dx = np.gradient(vx, dx)
    dvy_dy, dvy_dx = np.gradient(vy, dx)

    # raster images have y+ down the page, so do we need to:
    dvx_dy *= -1.0
    dvy_dy *= -1.0

    xy_shear = (dvx_dy + dvy_dx) * 0.5

    shape1 = dvx_dx.shape[0]
    shape2 = dvx_dx.shape[1]

    # Reshape the strains for eigenvector calculation
    strain_hor = np.reshape(np.stack((dvx_dx,
                                      xy_shear,
                                      xy_shear,
                                      dvy_dy), axis=2),
                            newshape=(shape1, shape2, 2, 2))

    # Do the eigenvector calculation to get principal strains
    eig, __ = np.linalg.eigh(strain_hor)

    # Mask according to both gradient directions
    # Taking gradients in x & y directions results
    # in different missing values, so combine and mask
    mask = (dvx_dy.mask | dvx_dx.mask)
    eig[mask] = np.nan

    return eig

def poly_inside_bbox(geom, bbox):
    """Use a bounding box to cut a shapely geometry"""
    geoms = shapely.ops.split(geom, bbox.boundary)
    geom = [g for g in geoms if bbox.contains(g)]
    assert(len(geom) == 1)
    return geom[0]

def generate_boundary(mask, transform, simplify_tol=None, bbox=None):
    """
    Produce a GMSH-ready glacier boundary from a categorical raster ('mask')

    Uses rasterio to polygonize the raster, shapely to simplify calving fronts

    Arguments:

    mask: the categorical raster describing ice/rock/ocean etc
    transform: tuple defining the array row/col -> real coords transform
    simplify_tol: how much to simplify the rastery edges of the calving front

    Returns:

    full_ring:   a Shapely LinearRing
    ocean_labels: associated tags for 'Physical' ocean boundaries.
    ice_labels: associated tags for 'Physical' ice boundaries.
    """
    if bbox:
        assert isinstance(bbox, dict)
        bbox_poly = shapely.geometry.box(bbox['xmin'],
                                         bbox['ymin'],
                                         bbox['xmax'],
                                         bbox['ymax'])
        bbox_outline = bbox_poly.boundary

    # rasterio is particular about datatype
    mask = mask.astype(np.int32)

    if simplify_tol is None:
        simplify_tol = transform[0]  # raster pixel size threshold

    # Get polgon 'features' from raster
    feats = rasterio.features.shapes(mask, transform=transform)

    # Put this into a Shapely-ready dict
    results = list(
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(feats))

    # To Shapely (only keep polygon which has raster value '1' - i.e. it's ice)
    # hardcoded ice mask = 1
    ice_geom = [shape(x['geometry']) for x in results
                if x['properties']['raster_val'] == 1.0]

    # In case there are multiple geoms, one ought to be MUCH larger than the others
    # So just take this one
    areas = [ig.area for ig in ice_geom]
    idx = areas.index(max(areas))
    ice_geom = ice_geom[idx]

    # Take only the inside part
    if bbox:
        ice_geom = poly_inside_bbox(ice_geom, bbox_poly)

    ice_geom = ice_geom.exterior

    # But use these ones to determine ice/ocean edge
    # hardcoded ocean mask = 0
    ocean_geoms = [shape(x['geometry']) for x in results
                   if x['properties']['raster_val'] == 0.0]

    # Clear out tiny ocean_geoms in slightly noisy bedmachine:
    ocean_geoms = [og for og in ocean_geoms if not Polygon(ice_geom).contains(og)]

    # Take only the inside part
    if bbox:
        ocean_geoms = [poly_inside_bbox(og, bbox_poly) for og in ocean_geoms]

    ocean_geoms = [og.exterior for og in ocean_geoms]

    # Separate the ice/exterior edges from ice/ocean
    ice_only = ice_geom
    for og in ocean_geoms:
        ice_only -= og

    # Get the ice/ocean sections (and simplify them)
    ocean_segs = []
    ocean_simp = []
    for og in ocean_geoms:
        intersect = shapely.ops.linemerge(og.intersection(ice_geom))
        ocean_segs.append(intersect)
        ocean_simp.append(intersect.simplify(tolerance=simplify_tol))

    all_bits = []
    all_bits.extend(ocean_simp)
    all_bits.extend([g for g in ice_only.geoms])

    # Merge all sections back into LinearRing
    full_ring = shapely.geometry.LinearRing(shapely.ops.linemerge(all_bits))

    assert full_ring.is_simple  # check no overlaps

    # TODO - generalize - what about massive nunataks?

    # Need to produce a list of 'physical' labels for each line segment in the ring
    # A LinearRing will have as many simple line segments as it has points
    # But it's Shapely representation has a duplicate point, so n-1

    npoints = len(full_ring.coords)
    nedges = npoints - 1
    ocean_labels = [[] for j in ocean_simp]  # 1 group per calving bit
    ice_labels = []

    # Gather lists of edges on ocean boundaries (for GMSH Physical Lines)
    # Edge number n connects points n and n+1
    for i in range(nedges):
        isocean = False
        edge = shapely.geometry.LineString(full_ring.coords[i:i+2])
        for j, og in enumerate(ocean_simp):
            if og.contains(edge):
                isocean = True
                ocean_labels[j].append(i+1)  # gmsh is 1 indexed
                break
        if not isocean:
            ice_labels.append(i+1)

    assert sum([len(e) for e in ocean_labels]) + len(ice_labels) == nedges

    for line in ocean_labels:
        assert len(line) > 0
    assert len(ice_labels) > 0

    return full_ring, ice_labels, ocean_labels

def map_lines_to_tris(line_cells, tri_cells):
    """
    Return a mapping from line elements to parent triangles.

    See UFC Specification and User Manual v. 1.1 for
    local numbering convention

    Arguments:

    line_cells - array of vertex IDs defining each line
    tri_cells  - array of vertex IDs defining each tri

    Returns:

    elem_nos - the triangle in which each line belongs
    local_idx - the local numbering of the line within the tri
    """
    tri_cells = np.asarray(tri_cells)
    line_cells = np.asarray(line_cells)
    line_cells.sort(axis=1)

    nlines = line_cells.shape[0]

    # Check triangles, element node indices always monotonic
    assert tri_cells.shape[1] == 3
    assert np.all(tri_cells[:, 0] < tri_cells[:, 1])
    assert np.all(tri_cells[:, 1] < tri_cells[:, 2])

    elem_nos = np.zeros(nlines, dtype=np.int)
    local_idx = np.zeros(nlines, dtype=np.int)

    # Lines are number by the vertex they *don't* contain
    idx_map = [(0, 1, 2),
               (0, 2, 1),
               (1, 2, 0)]

    # Where do we find this line in the tris?
    # NB: FEniCS has a mesh.topology() tool but I wasn't able
    # to achieve the same result, so rolled my own.
    for i in range(nlines):
        for j in range(3):
            result = np.where((tri_cells[:, idx_map[j][:2]] ==
                               line_cells[i, :]).all(axis=1))[0]
            if len(result) > 0:
                elem_nos[i] = result[0]
                local_idx[i] = idx_map[j][2]
                break

    return elem_nos, local_idx


def write_mvc(mvc, fname):
    """Write a MeshValueCollection to file"""
    mvc_xdmf = df.XDMFFile(df.MPI.comm_world, str(fname))
    mvc_xdmf.write(mvc)

def init_mvc(fenics_mesh):
    """Initialise an empty MeshValueCollection based on a fenics mesh"""
    return df.MeshValueCollection("size_t", fenics_mesh, dim=1)

def lines_to_mvc(inmesh, fenics_mesh, marker_name="gmsh:physical"):
    """Extract the BC markers & return them as a MeshValueCollection"""
    # Where do we find the line data?
    line_idx = [i for i, c in enumerate(inmesh.cells) if c.type == 'line']

    # Get the line elements & their markers
    line_cells = []
    func_vals = []
    for idx in line_idx:
        line_cells.extend(inmesh.cells[idx].data)
        func_vals.extend(inmesh.cell_data[marker_name][idx])

    # and (triangular) fenics mesh cells
    fcells = fenics_mesh.cells()

    assert len(func_vals) == len(line_cells)

    # Work out the mapping from lines to tris
    elem_nos, local_idx = map_lines_to_tris(line_cells, fcells)

    mvc = init_mvc(fenics_mesh)
    for elem, idx, val in zip(elem_nos, local_idx, func_vals):
        mvc.set_value(elem, idx, val)

    return mvc

def get_netcdf_coords(netcdf_df):
    """Return x and y coords from netcdf"""
    xx = netcdf_df['xaxis'][:]
    yy = netcdf_df['yaxis'][:]

    # Unmask if necessary
    if isinstance(xx, np.ma.core.MaskedArray):
        xx = xx.data
        yy = yy.data

    return xx, yy

def write_medit_sol(metric, filename):
    """Write a field defined at mesh nodes to medit .sol file"""
    n = metric.shape[0]

    with open(filename, 'w') as sol:
        sol.write("MeshVersionFormatted 2\n\n")
        sol.write("Dimension 2\n\n")
        sol.write("SolAtVertices\n")
        sol.write(f"{n}\n")
        sol.write("1 1\n\n")
        for i in range(n):
            sol.write("%f\n" % metric[i])

def build_gmsh_domain(domain_ring, ice_labels, ocean_labels, lc=1000.0):
    """
    Initialize gmsh api and create the domain

    Mesh is uniform lc, for interpolating metric prior to MMG refinement.

    Arguments:

    domain_ring - Shapely LinearRing defining the domain exterior
    ice_labels - list of lines which define ice BCs
    ocean_labels - list of lists of lines which define ocean BCs

    Returns:

    ice_tag - the tag of the ice boundary sections
    ocean_tags - the tags of the calving fronts
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("Glacier")

    # Load points & lines from shapely polygon
    x, y = domain_ring.xy

    # last point repeated
    assert x[0] == x[-1]
    assert y[0] == y[-1]
    x = x[:-1]
    y = y[:-1]

    npts = len(x)

    for i, (xx, yy) in enumerate(zip(x, y)):
        print(f"Point {i+1}")
        gmsh.model.geo.addPoint(xx, yy, 0, lc, i+1)  # gmsh 1-indexed

    for i in range(1, npts+1):
        print(f"Line {i} from {i} to  {(i % npts)+1}")
        gmsh.model.geo.addLine(i, (i % npts)+1, i)

    gmsh.model.geo.synchronize()

    loop = gmsh.model.geo.addCurveLoop([i for i in range(1, npts+1)], 1)
    surf = gmsh.model.geo.addPlaneSurface([loop], 1)

    ice_tag = gmsh.model.addPhysicalGroup(dim=1, tags=ice_labels, tag=1)

    # TODO - predefine tags so we know which are ocean/ice? Output this info (not here)
    ocean_tags = [gmsh.model.addPhysicalGroup(dim=1, tags=label, tag=-1) for
                  i, label in enumerate(ocean_labels)]

    # Add the (only) physical body (is this necessary?)
    gmsh.model.addPhysicalGroup(dim=2, tags=[1], tag=1)

    gmsh.model.geo.synchronize()

    return ice_tag, ocean_tags

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

def extract_tri_mesh(inmesh):
    """Given a mixed element mesh, produce the equivalent triangle only mesh"""
    # Cycle the cell blocks looking for triangles
    for c in inmesh.cells:
        if c.type == 'triangle':
            cells = c

    assert cells is not None

    outmesh = meshio.Mesh(points=inmesh.points,
                          cells=[cells])

    return outmesh

def load_fenics_mesh(fenics_meshfile):
    """Get a fenics mesh in XDMF format"""
    # Load the fenics mesh back in so we can construct the MVC
    mesh_in = df.Mesh()
    mesh_xdmf = df.XDMFFile(df.MPI.comm_world, str(fenics_meshfile))
    mesh_xdmf.read(mesh_in)

    return mesh_in

def simple_strain_metric(field, lc_params):
    """Turn sum of eigenstrains into a edge length metric"""
    metric = lc_params['strain_scale'] / field

    metric = np.clip(metric, lc_params['min'], lc_params['max'])
    metric[np.isnan(metric)] = lc_params['max']

    return metric

def run_mmg_adapt(mesh_fname, sol_fname, hgrad, hausd):
    """
    Use MMG to adapt mesh to target metric

    Low hausd means very high mesh resolution near calving front.

    Arguments:

    mesh_fname - the filename for the input mesh.
    sol_fname  - the filename for the solution (metric).
    hgrad      - maximum edge-length ratio of neighbouring elements (float)
    hausd      - max distance by which edges can be moved from original
    """
    assert MMG_LIB_LOC is not None
    my_env = os.environ.copy()
    # if current set up does not wokr try the line below with 
    # MMG_LIB_LOC_PATH = /usr/bin/, being the absolute path location of the mmg binary
    # in my case /home/brecinos/bin
    # my_env["LD_LIBRARY_PATH"] = MMG_LIB_LOC_PATH + ":" + my_env["LD_LIBRARY_PATH"]
    subprocess.run([MMG_LIB_LOC,
                    str(mesh_fname),
                    "-sol", str(sol_fname),
                    "-hgrad", str(hgrad),
                    "-hausd", str(hausd)],
                   env=my_env) 

def remove_medit_corners(infile):
    """Get rid of 'Corners' section of medit mesh, which meshio can't handle"""
    temp = tempfile.TemporaryFile(mode='w+')
    triggered = False
    with open(infile, 'r') as inny:
        for line in inny:
            if triggered:
                if line == 'Edges\n':
                    triggered = False
            elif 'Corners' in line:
                triggered = True

            if not triggered:
                temp.writelines(line)

    temp.seek(0)
    lines = temp.readlines()

    with open(infile, 'w') as outy:
        outy.writelines(lines)

def interp_to_gmsh(metric, metric_xx, metric_yy):
    """Take the metric from netcdf grid and interp it onto our gmsh mesh"""

    node_tags, node_xyz, _ = gmsh.model.mesh.getNodes()
    node_xyz = node_xyz.reshape(-1, 3)

    interper = RegularGridInterpolator((metric_xx, metric_yy), metric.T)
    interped_metric = interper(node_xyz[:, :2])

    return interped_metric

def gmsh_to_medit(infile, outfile):
    """
    Convert gmsh to medit, converting 'physical' boundaries to cell_data

    meshio is *very* particular about the format of this stuff, and also
    at the v4 release, changed a lot of stuff. Even the changelogs claim
    that the 'cell_data' ought to be a dict of lists, but actually only a
    dict of numpy arrays works!

    So - cells is a dict of lists, one per element type
      cell_data is a dict of lists of numpy arrays!

    Arguments:

    infile - the gmsh filename to be converted
    outfile - the medit filename for the result
    """

    inmesh = meshio.read(infile)

    gmsh_phys = inmesh.cell_data["gmsh:physical"]
    gmsh_phys_lines = np.hstack(gmsh_phys[:-1])
    gmsh_phys_tris = gmsh_phys[-1]
    gmsh_phys_list = [gmsh_phys_lines, gmsh_phys_tris]

    outmesh = meshio.Mesh(points=inmesh.points[:, :2],
                          cells={"line": inmesh.cells_dict["line"],
                                 "triangle": inmesh.cells_dict["triangle"]},
                          cell_data={"gmsh:physical": gmsh_phys_list})

    meshio.write(outfile, outmesh)

def tags_to_file(tag_dict, outfile):
    """Write boundary tags to a text file"""
    with open(outfile, 'w') as output:
        for key in tag_dict:
            for v in tag_dict[key]:
                output.write(f"{key}: {v}\n")

def delete_intermediates(name_root):
    print(name_root)
    delete_exts = [".o.sol", ".sol", ".o.mesh", ".mesh", ".msh"]
    for d in delete_exts:
        os.remove(name_root + d)

def slice_by_xy(arr, xx, yy, extent):
    """Return the section of an array (and xx, yy) which are within bounds"""

    assert arr.shape == (yy.shape[0], xx.shape[0])
    x_inds = np.where((xx >= extent["xmin"]) & (xx <= extent["xmax"]))[0]
    y_inds = np.where((yy >= extent["ymin"]) & (yy <= extent["ymax"]))[0]

    # surely there's a better way to use np.where here?
    sliced_arr = arr[y_inds[0]:y_inds[-1]+1, x_inds[0]:x_inds[-1]+1]

    return sliced_arr, xx[x_inds], yy[y_inds]

def slice_netcdf(netcdf_df, varname, extent, x_varname='x', y_varname='y', return_transform=True):
    """Return a slice of a netcdf variable given bounds"""
    xx = netcdf_df[x_varname][:]
    yy = netcdf_df[y_varname][:]
    x_inds = np.where((xx >= extent["xmin"]) & (xx <= extent["xmax"]))[0]
    y_inds = np.where((yy >= extent["ymin"]) & (yy <= extent["ymax"]))[0]

    dims = netcdf_df[varname].dimensions
    assert set(dims) == set((x_varname, y_varname))

    sliced_var = netcdf_df[varname][y_inds, x_inds].data
    if dims[0] == x_varname:
        sliced_var = sliced_var.T

    xx = xx[x_inds]
    yy = yy[y_inds]

    # Check origin top-left
    assert (yy[0] - yy[-1] > 0)
    assert (xx[0] - xx[-1] < 0)

    dy = abs(yy[0] - yy[1])
    dx = abs(xx[0] - xx[1])

    # Pixel corner
    origin_y = yy[0] + dy*0.5
    origin_x = xx[0] - dx*0.5

    if return_transform:
        # Use rasterio (origin -> transformation)
        affine_transform = rasterio.transform.from_origin(west=origin_x, north=origin_y,
                                                      xsize=dx, ysize=dy)

        return sliced_var, affine_transform

    else:

        return sliced_var, xx, yy

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

def check_if_arrays_have_same_shape(arrays, array_shape):
    """
    Returns a list of bools checking if all arrays have the same
    shape
    :param arrays: list of arrays to check
    :param array_shape: shape to check against
    :return: a list of True or False
    """
    bools = []
    for array in arrays:
        if array.shape == array_shape:
            bools.append(True)
        else:
            bools.append(False)
    return bools

def crop_velocity_data_to_extend(dvel, extend,
                                 return_coords=False,
                                 return_indexes=False,
                                 return_xarray=False):
    """
    Returns a xarray.Dataset crop to the given
    extent
    :param
        dvel: xarray.Dataset to crop
        extend: extend to crop the data to
        given as a dictionary with the form:
        e.g. {'xmin': -1609000.0, 'xmax': -1381000.0,
              'ymin': -718450.0, 'ymax': -527000.0}
        return_coords: Bool to return x and y coordinates
        default is False
    :return
        dv.data: numpy array with data
        dv.x.data: x coordinates
        dv.y.data: y coordinates
    """

    # Processing vel data
    x_coords = dvel.x.data
    y_coords = dvel.y.data

    x_inds = np.where((x_coords >= extend["xmin"]) & (x_coords <= extend["xmax"]))[0]
    y_inds = np.where((y_coords >= extend["ymin"]) & (y_coords <= extend["ymax"]))[0]

    dv = dvel.isel(x=x_inds, y=y_inds)

    if return_xarray and return_indexes:
        return dv, x_inds, y_inds
    elif return_xarray:
        return dv
    elif return_coords:
        return dv.data, dv.x.data, dv.y.data
    else:
        return dv.data

def interpolate_missing_data(array, xx, yy):
    """
    Interpolate missing data via interpolate.gridata
    using nearest neighbour
    :param array: numpy array
    :param xx: 2d coord
    :param yy: 2d coord
    :return: flatten array
    """
    # mask invalid values
    array_ma = np.ma.masked_invalid(array)

    # get only the valid values
    x1 = xx[~array_ma.mask]
    y1 = yy[~array_ma.mask]
    newarr = array[~array_ma.mask]

    array_int = griddata((x1, y1), newarr.ravel(),
                   (xx, yy),
                   method='nearest')

    return array_int.ravel()

def open_and_crop_itslive_data(path_to_data, extend,
                               x_to_int, y_to_int):
    """
    Opens and crops itslive data to the smith Glacier domain
    and puts them in the same grid as ASE

    :param path_to_data: string to itslive files
    :param extend: model domain extend
    :param x_to_int: x coordinate to interpolate to
    :param y_to_int: y coordinate to interpolate to
    :return: velocity components and uncertainty in the smith
    glacier domain and with the same resolution than ASE

    """
    dvel = xr.open_dataset(path_to_data)

    vel_obs = dvel.v
    vx = dvel.vx
    vy = dvel.vy
    vx_err = dvel.vx_err
    vy_err = dvel.vy_err

    # Crop to smith Glacier extend
    vx_s, xind, yind = crop_velocity_data_to_extend(vx, extend, return_xarray=True, return_indexes=True)
    vy_s = crop_velocity_data_to_extend(vy, extend, return_xarray=True)
    errx_s = crop_velocity_data_to_extend(vx_err, extend, return_xarray=True)
    erry_s = crop_velocity_data_to_extend(vy_err, extend, return_xarray=True)

    vel_obs_s = crop_velocity_data_to_extend(vel_obs, extend, return_xarray=True)

    # Interpolate ITSlive 2011 data to the same resolution of ASE
    vx_int = vx_s.interp(x=x_to_int, y=y_to_int)
    vy_int = vy_s.interp(x=x_to_int, y=y_to_int)
    errx_int = errx_s.interp(x=x_to_int, y=y_to_int)
    erry_int = erry_s.interp(x=x_to_int, y=y_to_int)

    vel_obs_int = vel_obs_s.interp(x=x_to_int, y=y_to_int)

    return vel_obs_int, vx_int, vy_int, errx_int, erry_int

def drop_invalid_data_from_several_arrays(x, y,
                                          vx, vy,
                                          vx_err, vy_err, masked_array):
    """

    :param x: 1d array of coordinates to for a grid
    :param y: 1d array of coordinates to for a grid
    :param vx: velocity component in x direction
    :param vy: velocity component in y direction
    :param vx_err: velocity component error in x direction
    :param vy_err: velocity component error in y direction
    :param masked_array: numpy.mask array
    :return: variables without nans
    """

    # Now we drop invalid data
    x_grid, y_grid = np.meshgrid(x, y)

    x_nonan = x_grid[~masked_array.mask]
    y_nonan = y_grid[~masked_array.mask]

    vel_nonan = masked_array.data[~masked_array.mask]
    vy_nonan = vy[~masked_array.mask]
    vx_nonan = vx[~masked_array.mask]

    vy_err_nonan = vy_err[~masked_array.mask]
    vx_err_nonan = vx_err[~masked_array.mask]

    shape_after = vy_nonan.shape
    print('Shape after nan drop mus be values,')
    print(shape_after)

    all_data = [x_nonan, y_nonan,
                vel_nonan, vy_nonan, vx_nonan,
                vy_err_nonan, vx_err_nonan]

    bool_list = check_if_arrays_have_same_shape(all_data,
                                                shape_after)

    assert all(element == True for element in bool_list)
    return x_nonan, y_nonan, vx_nonan, vy_nonan, vx_err_nonan, vy_err_nonan, vel_nonan

def get_data_for_experiment(path, experiment_name):
    """
    Read .csv files for each l-curve experiment and append
    all results in a single pandas data frame

    :param path to experiment file
    :param experiment name e.g. gamma_alpha, gamma_beta
    :return: pandas.Dataframe with the inv J_cost function results
    for each l-curve experiment.
    """
    j_paths_small = []
    j_paths_big = []
    dir_path = os.path.join(path, experiment_name)
    for root, dirs, files in os.walk(dir_path):
        dirs.sort()
        files = [os.path.join(root, f) for f in files]
        excludes = ['*inversion_progress*', '*1e+*', '*.xml', '*.h5', '*.xdmf']
        excludes = r'|'.join([fnmatch.translate(x) for x in excludes]) or r'$.'
        j_paths_small = [f for f in files if not re.match(excludes, f)]
        excludes = ['*inversion_progress*', '*1e-*', '*.xml', '*.h5', '*.xdmf']
        excludes = r'|'.join([fnmatch.translate(x) for x in excludes]) or r'$.'
        j_paths_big = [f for f in files if not re.match(excludes, f)]

    j_paths_small.reverse()
    j_paths = j_paths_small + j_paths_big

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
    if add_mesh:
        ax.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
    minv = vmin
    maxv = vmax
    levels = np.linspace(minv, maxv, 200)
    ticks = np.linspace(minv, maxv, 3)
    c = ax.tricontourf(x, y, t, field, levels=levels, cmap=cmap)
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

def compute_vertex_for_dQ_components(Q, dQ, mesh, hd5_fpath=str, n_sens=int, mult_mmatrix=False):

    hdf5data = HDF5File(MPI.comm_world, hd5_fpath, 'r')
    hdf5data.read(dQ, f'dQdalphaXbeta/vector_{n_sens}')

    dx = Measure('dx', domain=mesh)
    Qp_test, Qp_trial = TestFunction(Q), TrialFunction(Q)

    # Mass matrix solver
    M_mat = assemble(inner(Qp_trial, Qp_test) * dx)

    from dolfin import KrylovSolver
    M_solver = KrylovSolver(M_mat.copy(), "cg", "sor")  # DOLFIN KrylovSolver object
    M_solver.parameters.update({"relative_tolerance": 1.0e-14,
                                "absolute_tolerance": 1.0e-32})

    from tlm_adjoint.interface import function_new
    this_action = function_new(dQ, name=f"M_inv_action")

    M_solver.solve(this_action.vector(), dQ.vector())

    if mult_mmatrix:
        dQ_alpha, dQ_beta = this_action.split(deepcopy=True)
    else:
        dQ_alpha, dQ_beta = dQ.split(deepcopy=True)

    # Vector to plot
    va = dQ_alpha.compute_vertex_values(mesh)
    vb = dQ_beta.compute_vertex_values(mesh)

    return va, vb
