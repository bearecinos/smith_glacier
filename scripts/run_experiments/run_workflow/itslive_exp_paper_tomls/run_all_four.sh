#!/bin/bash

# Create and define your run_input directory for the workflow run
input_run_inv=$INPUT_DIR/input_run_workflow
if [ ! -d $input_run_inv ]
then
  echo "Creating run input directory $input_run_inv"
  mkdir $input_run_inv
else
  echo "Directory is $input_run_inv already exist"
fi

# Create output directory for inversion output
export run_inv_output_dir=$OUTPUT_DIR/06_workflow_prios
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

toml set --toml-path $2 io.input_dir "$input_run_inv"
toml set --toml-path $2 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $2 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

toml set --toml-path $3 io.input_dir "$input_run_inv"
toml set --toml-path $3 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $3 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

toml set --toml-path $4 io.input_dir "$input_run_inv"
toml set --toml-path $4 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $4 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

toml set --toml-path $5 io.input_dir "$input_run_inv"
toml set --toml-path $5 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $5 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

echo $(date -u) "Run started"

mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $RUN_CONFIG_DIR/run_workflow/$2
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $RUN_CONFIG_DIR/run_workflow/$3
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $RUN_CONFIG_DIR/run_workflow/$4
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $RUN_CONFIG_DIR/run_workflow/$5

echo $(date -u) "Done with Inversion!"

mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_forward.py $RUN_CONFIG_DIR/run_workflow/$2
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_forward.py $RUN_CONFIG_DIR/run_workflow/$3
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_forward.py $RUN_CONFIG_DIR/run_workflow/$4
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_forward.py $RUN_CONFIG_DIR/run_workflow/$5

echo $(date -u) "Done with Forward!"

mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_eigendec.py $RUN_CONFIG_DIR/run_workflow/$2
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_eigendec.py $RUN_CONFIG_DIR/run_workflow/$3
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_eigendec.py $RUN_CONFIG_DIR/run_workflow/$4
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_eigendec.py $RUN_CONFIG_DIR/run_workflow/$5

echo $(date -u) "Done with eigendec -------------------------------------------------------!"

mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_errorprop.py $RUN_CONFIG_DIR/run_workflow/$2
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_errorprop.py $RUN_CONFIG_DIR/run_workflow/$3
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_errorprop.py $RUN_CONFIG_DIR/run_workflow/$4
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_errorprop.py $RUN_CONFIG_DIR/run_workflow/$5

echo $(date -u) "Done with error propagation"
