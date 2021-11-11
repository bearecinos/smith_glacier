# Smith Glacier experiments

This repository consist of several experiments with [Fenics_ice](https://github.com/EdiGlacUQ/fenics_ice) over Smith, Pope, and Kohler Glaciers. They are three narrow ∼ 10 km wide interconnected West Antarctic ice streams.

> **Note**: this repository and its documentation is a work in progress :construction_worker: 

The code has been developed to: 

1. Create a finite element mesh of the region of interest, where simulations will be carried out.
2. Generate the all input data (crop to the study domain) needed by Fenics_ice.

And to carry out the following experiments with the model:

3. Invert for basal drag and ice stiffness by minimizing the difference between model and observed ice velocities (*J<sup>c</sup><sub>mis</sub>*).
4. Run the eigendecomposition of the Hessian matrix of the model-observations misfit *J<sup>c</sup><sub>mis</sub>* and multiply this, by the inverse of the covariance matrix of the basal drag/ice stiffness. 
5. And finally project this covariance on a linearization of the time-dependent ice sheet model (using Automatic Differentiation to generate the linearization) and estimate the growth of a QoI (Quantity of Interest) uncertainty over time (e.g. Ice mass loss).


Repository structure:
---------------------

- `meshtools`: A set of tools for mesh and input data generation.
- `scripts`: Python and bash scripts to run each data processing stage.
   - `prepro_stages`: Python scripts to generate the input data for the model runs cropped to the study region.
   - `run_stages`: Bash scritps and `.toml` configuration files for each run stage of fenics_ice.

- `config.init`: Configuration file for paths to input and output data.
- `setpaths.sh`: Global paths to repository and repository directories.

Installation and usage
----------------------

### Check out the repository [Wiki](https://github.com/bearecinos/smith_glacier/wiki#welcome-to-the-documentation-website-for-smith_glacier) for installation and usage.
