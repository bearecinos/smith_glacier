"""
Takes a Mesh and MeshValueCollection (created by e.g. smith_mesh_metric_mmg.py) and
writes out a VTK to inspect it in paraview.
"""
import os
import sys
from configobj import ConfigObj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

MAIN_PATH = config['main_path']
sys.path.append(MAIN_PATH)

from ficetools.backend import MPI, Mesh, MeshFunction, MeshValueCollection, XDMFFile, File

output_path = os.path.join(MAIN_PATH,
                            'output/01_mesh')

meshname =  os.path.join(output_path, 'smith_variable_ocean')
meshfile = os.path.join(MAIN_PATH, config['meshfile'])
mvc_file = os.path.join(MAIN_PATH, config['mvc_file'])

mesh_in = Mesh()
mesh_xdmf = XDMFFile(MPI.comm_world, str(meshfile))
mesh_xdmf.read(mesh_in)


mvc = MeshValueCollection("size_t", mesh_in, dim=1)
mvc_xdmf = XDMFFile(MPI.comm_world, str(mvc_file))
mvc_xdmf.read(mvc)

mfunc = MeshFunction('size_t', mesh_in, mvc)

vtk_fname = meshname+"_ff.pvd"
File(vtk_fname) << mfunc
