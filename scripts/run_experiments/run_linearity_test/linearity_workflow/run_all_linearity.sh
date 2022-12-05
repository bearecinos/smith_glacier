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
export run_inv_output_dir=$OUTPUT_DIR/11_linearity_workflow
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

path_logs=$RUN_CONFIG_DIR/run_linearity_test/linearity_workflow
echo "Logs will be store here:" $path_logs

toml set --toml-path $2 io.input_dir "$input_run_inv"
toml set --toml-path $2 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $2 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

toml set --toml-path $3 io.input_dir "$input_run_inv"
toml set --toml-path $3 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $3 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

echo $(date -u) "Run inversion stages started"

mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $2 |& tee $path_logs/log_tom2_inv.txt
OUT=$(tail "$path_logs/log_tom2_inv.txt")
echo $OUT | mail -s "run inv finish config1" beatriz.recinos@ed.ac.uk

mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $3 |& tee $path_logs/log_tom3_inv.txt
OUT=$(tail "$path_logs/log_tom3_inv.txt")
echo $OUT | mail -s "run inv finish config2" beatriz.recinos@ed.ac.uk

#echo $(date -u) "Run forward linearity stage started"
#mpirun -n $1 python $RUN_CONFIG_DIR/run_linearity_test/run_linearity_fwd.py $2 $3 |& tee $path_logs/log_tom2_fwd.txt
#OUT=$(tail "$path_logs/log_tom2_fwd.txt")
#echo $OUT | mail -s "run fwd finish config1" beatriz.recinos@ed.ac.uk

#nohup bash $RUN_CONFIG_DIR/run_linearity_test/linearity_workflow/run_all_linearity.sh 24 $RUN_CONFIG_DIR/run_linearity_test/linearity_workflow/smith_itslive-std-original-complete_C0a2-8_L0a-3200_C0b2-28_L0b-1000.toml $RUN_CONFIG_DIR/run_linearity_test/linearity_workflow/smith_measures-std-original-complete_C0a2-8_L0a-3200_C0b2-28_L0b-1000.toml >/dev/null 2>&1
