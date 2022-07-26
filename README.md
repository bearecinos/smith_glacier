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

- `ficetools`: A set of tools for mesh and input data generation.
- `scripts`: Python and bash scripts to run each data processing stage.
   - `prepro_stages`: Python scripts to generate the input data cropped to the study region for all the model stages.
   - `run_experiments`: Bash scrips and `.toml` configuration files for each run experiment and stage of fenics_ice.
   - `plot_stages`: An alternative to Paraview to visualize the model input and output data with matplotlib.
- `config.ini`: Configuration file for paths to input and output data.
- `setpaths.sh`: Global paths to repository and repository directories.

Installation and usage
----------------------

### Check out the repository [Wiki](https://github.com/bearecinos/smith_glacier/wiki#welcome-to-the-documentation-website-for-smith_glacier) for installation and usage.


DNG: Addition to allow automating of runs
-----------------------------------------

New scripts have been added to allow automation. These are very specific to runs that i have been carrying out, code would need cleaning/generalisation to be merged. There is also almost no error checking!

Relevant files in `scripts/run_experiments/run_workflow`:
- `run_all.py`
- `run_all_auto.sh`
- `run_inv_auto.sh`
- `run_forward_auto.sh`
- `run_eigendec_auto.sh`
- `run_errorprop_auto.sh`

To begin, a binary "parameter file" needs to be created. This will be an `n` by 6 numpy array where `n` is the number of experiments. There are 6 parameters, which appear in this order:
- Alpha length scale (meters)
- Beta length scale (meters)
- Alpha local standard deviation (units not given)
- Beta local standard deviation (units not given)
- Step number in velocity data set (should be integer)
- Flag to either use MEaSUREs-based stdx/y adjustment (1 if used, 0 if not)

Example creation of an array:
~~~
In [1]: import numpy as np
In [2]: runs = np.empty(2,6)
In [3]: runs[:,:2] = 1000. # set Alpha/Beta length scale to 1000
In [4]: runs[:,2] = 100. # set Alpha C0**(1/2) = 100
In [5]: runs[:,3] = 50. # set Beta C0**(1/2) = 50
In [6]: runs[:,5] = 1. # adjusted stdev
In [7]: runs[0,4] = 4; runs[1,4] = 6 # set step sizes
In [8]: np.save('runs.npy',runs)
~~~

Then we can call `run_all.py`. The syntax is:
`python run_all.py [params_file] [script]`

If we use the parameter files above and only wish to run inversions, for instance, we call
`python run_all.py runs.py run_inv_auto`. Note that we leave off the `.sh` from the script name.

Once called, the script will use `os.system()` to call a shell script with the `nohup` command. In addition to running the `fenics_ice` experiment stage(s), the script will do the following:
- generate a **new** .toml file within `scripts/run_experiments/run_workflow/itslive_tomls`, with a name giving the details of the run. The toml uses `smith_cloud_subsampling.toml` as a template.
- create a **new** output folder within `$OUTPUT_DIR`, with a name giving the details of the run. The prior parameters here are in terms of gamma and delta, though.
- after each `fenics_ice` stage is complete, call the appropriate plotting script, with a subdirectory name based on the details of the run.
- pipe the stderr and stdout of each stage into a file in the top level of the output folder, so progress can be tracked and errors can be seen. 
- For the 1st run defined in `runs.npy`, the toml would be `smith_cloud_subsamplingstep_4_La_1000.0_Ca_100.0_Lb_1000.0_Cb_30.0_std_1.toml`. The output directory would be `$OUTPUT_DIR/03_step_4_ga_2.8_da_2.8e-06_gb_9.4_db_9.4e-06_std_1`. `stdout` from the inversion would appear in `$OUTPUT_DIR/03_step_4_ga_2.8_da_2.8e-06_gb_9.4_db_9.4e-06_std_1/out.inv`. Plots would appear in `$MAIN_PATH/exp_step_4_La_1000.0_Ca_100.0_Lb_1000.0_Cb_30.0_std_1`.
- if all is successful, the script will finally call `run_all.py` with the appropriate command line arguments. With each successive call, the top row is removed from `runs.npy`, and the next parameter list is used.

NOTE -- error checking is very poor. The best way to know if things are running well is to look at `top`. If there is an issue, you might need to look in `nohup.out`, or in the `stdout` and `stderr` files from each `fenics_ice` stage.

NOTE -- by default, `run_all.py` is configured to use 30 processors. This must be decreased ot 24 on bow.
