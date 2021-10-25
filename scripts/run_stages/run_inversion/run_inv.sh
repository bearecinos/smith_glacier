#!/bin/bash
# Copying data from output_prepro stages to input directory to be used for run stages
# Mesh files
cp $OUTPUT_DIR/01_mesh/smith_variable_ocean.xdmf $INPUT_DIR/input_runs/.
cp $OUTPUT_DIR/01_mesh/smith_variable_ocean_ff.xdmf $INPUT_DIR/input_runs/.
# Gridded data files
cp $OUTPUT_DIR/02_gridded_data/smith_bedmachine.h5 $INPUT_DIR/input_runs/.
cp $OUTPUT_DIR/02_gridded_data/smith_smb.h5 $INPUT_DIR/input_runs/.
cp $OUTPUT_DIR/02_gridded_data/smith_bglen.h5 $INPUT_DIR/input_runs/.
cp $OUTPUT_DIR/02_gridded_data/smith_obs_vel.h5 $INPUT_DIR/input_runs/.

# Create output directory for inversion output
export $run_inv_output_dir=$OUTPUT_DIR/03_run_inv
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

mpirun -n 12 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $RUN_CONFIG_DIR/run_inversion/smith.toml
