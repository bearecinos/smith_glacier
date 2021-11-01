# smith_glacier

Experiments with [Fenics_ice](https://github.com/EdiGlacUQ/fenics_ice) over Smith, Pope, and Kohler Glaciers. They are three narrow ∼ 10 km wide, interconnected West Antarctic ice streams.

> This repository and its documentation is a work in progress...

The code has been developed to: 

1. Create a finite element mesh for the region of interest where simulations will be carried out.
2. Generate the all input data (crop to the study domain) needed by Fenics_ice.

And also to carry out the following experiments with the model:

3. Invert for basal drag and ice stiffness by minimizing the difference between model and observed ice velocities.
4. Run the eigendecomposition of the Hessian matrix of the model-observations mistfit. This will allow us to multpli by the inverse of the covariance matrix of the basal drag/ice stiffness. 
5. Then the model will project the covariance on a linearization of the time-dependent ice sheet model (again using Automatic Differentiation to generate the linearization) to estimate the growth of QoI uncertainty over time (e.g. QoI is the ice mass loss).


Repository structure:

- `meshtools`: A set of tools for mesh and input data generation.
- `scripts`: Python and bash scripts to run each data processing stage.
- `config.init`: Configuration file for paths to input and output data.
- `setpaths.sh`: Global paths to repository and repository directories.

    

**Check out the repository [wiki]() for more information.**
