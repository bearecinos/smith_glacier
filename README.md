# Smith Glacier experiments

This repository consist of several experiments with [Fenics_ice](https://github.com/EdiGlacUQ/fenics_ice) over Smith, Pope, and Kohler Glaciers. They are three narrow ∼ 10 km wide interconnected West Antarctic ice streams.

> **Note**: this repository documentation is a work in progress :construction_worker: 

The code has been developed to: 

1. Create a finite element mesh of the region of interest, where simulations will be carried out.
2. Generate the all input data (crop to the study domain) needed by Fenics_ice.

And to carry out the following experiments with the model:

3. Invert for basal drag and ice stiffness by minimizing the difference between model and observed ice velocities (*J<sup>c</sup><sub>mis</sub>*).
4. Run the eigendecomposition of the Hessian matrix of the model-observations misfit *J<sup>c</sup><sub>mis</sub>* and multiply this, by the inverse of the covariance matrix of the basal drag/ice stiffness. 
5. And finally project this covariance on a linearization of the time-dependent ice sheet model (using Automatic Differentiation to generate the linearization) and estimate the growth of a QoI (Quantity of Interest) uncertainty over time (e.g. Ice mass loss).

Repository structure:
---------------------

- `ficetools`: inner python module that host various useful functions for the pre-processing stages, handling files and plotting routines.
- `scripts`: Python and bash scripts to run each data processing stage.
   - `prepro_stages`: Python scripts to generate the input data cropped to the study region for all the model stages.
   - `run_experiments`: Bash scrips and `.toml` configuration files for each run experiment and stage of fenics_ice.
   - `plot_stages`: An alternative to Paraview to visualize the model input and output data with matplotlib.
   - `post_processing`: Python scripts to re-grid the Fenics_ice output to a rectangular mesh.
- `config.ini`: Local configuration file for paths to input and output data.
- `setpaths.sh`: Global paths to repository and repository directories.

Installation and usage
----------------------

### Check out the repository [Wiki](https://github.com/bearecinos/smith_glacier/wiki#welcome-to-the-documentation-website-for-smith_glacier) for installation and usage.
