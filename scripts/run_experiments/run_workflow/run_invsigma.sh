#!/bin/bash

# Check that the right output folder and input folder exist
input_run_inv=$INPUT_DIR/input_run_inv
run_inv_output_dir=$OUTPUT_DIR/03_run_inv

if [ ! -d $input_run_inv ]
then
  echo "Creating run input directory $input_run_inv"
  mkdir $input_run_inv
  echo "Copying prepro output files to input run dir"
  # Mesh files
  cp $OUTPUT_DIR/01_mesh/smith_variable_ocean* $input_run_inv/.
  # Gridded data files
  cp $OUTPUT_DIR/02_gridded_data/smith_bedmachine.h5 $input_run_inv/.
  cp $OUTPUT_DIR/02_gridded_data/smith_smb.h5 $input_run_inv/.
  cp $OUTPUT_DIR/02_gridded_data/smith_bglen.h5 $input_run_inv/.
  cp $OUTPUT_DIR/02_gridded_data/smith_obs_vel_* $input_run_inv/.
  cp $OUTPUT_DIR/02_gridded_data/smith_melt_depth_params.h5 $input_run_inv/.
else
  echo "Directory $input_run_inv already exists, we dont need to copy anything"
fi

if [ ! -d $run_inv_output_dir ]
then
  echo "Directory $run_inv_output_dir DOES NOT exists, run first run_inv!"
else
  echo "Directory $run_inv_output_dir exist, we can run everything now"
  echo $(date -u) "Run started"
  mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_invsigma.py $RUN_CONFIG_DIR/run_workflow/$2
  echo $(date -u) "Done!"
fi
