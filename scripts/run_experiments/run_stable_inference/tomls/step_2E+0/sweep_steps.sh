# Create run_input directory for the inversion run
input_run_inv=$INPUT_DIR/input_run_workflow
if [ ! -d $input_run_inv ]
then
  echo "Creating run input directory $input_run_inv"
  mkdir $input_run_inv
else
  echo "Directory is $input_run_inv already exist"
fi

# Create output directory for inversion output
export run_inv_output_dir=$OUTPUT_DIR/14_stable_inference_test
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

script_inv=$RUN_CONFIG_DIR/run_stable_inference/run_stable_inference_inv.py
DIR_TOMLS=$RUN_CONFIG_DIR/run_stable_inference/tomls/step_2E+0

toml set --toml-path $DIR_TOMLS/smith_cloud_training_middle_step_2E+0_.toml io.input_dir "$input_run_inv"
toml set --toml-path $DIR_TOMLS/smith_cloud_training_zero_step_2E+0_.toml io.input_dir "$input_run_inv"
toml set --toml-path $DIR_TOMLS/smith_cloud_testing_middle_step_2E+0_.toml io.input_dir "$input_run_inv"
toml set --toml-path $DIR_TOMLS/smith_cloud_testing_zero_step_2E+0_.toml io.input_dir "$input_run_inv"

toml set --toml-path $DIR_TOMLS/smith_cloud_training_middle_step_2E+0_.toml io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $DIR_TOMLS/smith_cloud_training_zero_step_2E+0_.toml io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $DIR_TOMLS/smith_cloud_testing_middle_step_2E+0_.toml io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $DIR_TOMLS/smith_cloud_testing_zero_step_2E+0_.toml io.output_dir "$run_inv_output_dir/output"

toml set --toml-path $DIR_TOMLS/smith_cloud_training_middle_step_2E+0_.toml io.diagnostics_dir "$run_inv_output_dir/diagnostics"
toml set --toml-path $DIR_TOMLS/smith_cloud_training_zero_step_2E+0_.toml io.diagnostics_dir "$run_inv_output_dir/diagnostics"
toml set --toml-path $DIR_TOMLS/smith_cloud_testing_middle_step_2E+0_.toml io.diagnostics_dir "$run_inv_output_dir/diagnostics"
toml set --toml-path $DIR_TOMLS/smith_cloud_testing_zero_step_2E+0_.toml io.diagnostics_dir "$run_inv_output_dir/diagnostics"

mpirun -n 24 python $script_inv $DIR_TOMLS/smith_cloud_training_middle_step_2E+0_.toml $DIR_TOMLS/smith_cloud_testing_middle_step_2E+0_.toml
mpirun -n 24 python $script_inv $DIR_TOMLS/smith_cloud_training_zero_step_2E+0_.toml $DIR_TOMLS/smith_cloud_testing_zero_step_2E+0_.toml

echo $(date -u) "We are done with the stable inference for step 2E+0!"