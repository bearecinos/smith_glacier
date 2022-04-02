#!/bin/bash

# Create run_input directory for the inversion run
input_run_inv=$INPUT_DIR/input_run_inv
if [ ! -d $input_run_inv ]
then
  echo "Creating run input directory $input_run_inv"
  mkdir $input_run_inv
else
  echo "Directory is $input_run_inv already exist"
fi

echo "Copying prepro output files to input run dir"
# Mesh files
cp $OUTPUT_DIR/01_mesh/smith_variable_ocean* $input_run_inv/.
# Gridded data files
cp $OUTPUT_DIR/02_gridded_data/smith_bedmachine.h5 $input_run_inv/.
cp $OUTPUT_DIR/02_gridded_data/smith_smb.h5 $input_run_inv/.
cp $OUTPUT_DIR/02_gridded_data/smith_bglen.h5 $input_run_inv/.
cp $OUTPUT_DIR/02_gridded_data/smith_obs_vel_measures-comp.h5 $input_run_inv/.
cp $OUTPUT_DIR/02_gridded_data/smith_melt_depth_params.h5 $input_run_inv/.

# Create output directory for inversion output
export run_inv_output_dir=$OUTPUT_DIR/03_run_inv
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

echo $(date -u) "Run started"

mpirun -n 24 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $RUN_CONFIG_DIR/run_workflow/smith.toml

echo $(date -u) "Done!"

