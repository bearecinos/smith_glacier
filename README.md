# Smith Glacier experiments

Experiments with [Fenics_ice](https://github.com/EdiGlacUQ/fenics_ice) over Smith, Pope, and Kohler Glaciers. They are three narrow ∼ 10 km wide, interconnected West Antarctic ice streams.

> **This repository and its documentation is a work in progress!**

The code has been developed to: 

1. Create a finite element mesh of the region of interest, where simulations will be carried out.
2. Generate the all input data (crop to the study domain) needed by Fenics_ice.

And to carry out the following experiments with the model:

3. Invert for basal drag and ice stiffness by minimizing the difference between model and observed ice velocities.
4. Run the eigendecomposition of the Hessian matrix of the model-observations misfit. This will allow us to multiply by the inverse of the covariance matrix of the basal drag/ice stiffness. 
5. Then the model will project the covariance on a linearization of the time-dependent ice sheet model (again using Automatic Differentiation to generate the linearization) to estimate the growth of QoI uncertainty over time (e.g. Ice mass loss).

Repository structure:
---------------------

- `meshtools`: A set of tools for mesh and input data generation.
- `scripts`: Python and bash scripts to run each data processing stage.
   - `prepro_stages`: Python scripts to generate the input data for the model runs cropped to the study region.
   - `run_stages`: Bash scritps and `.toml` configuration files for each run stage of fenics_ice.

- `config.init`: Configuration file for paths to input and output data.
- `setpaths.sh`: Global paths to repository and repository directories.


**Check out the repository [Wiki]() for installation and usage.**
