#!/bin/bash
input_run_inv=$INPUT_DIR/input_run_inv

if [ ! -d $input_run_inv ]
then
  echo "Creating run input directory $input_run_inv"
  mkdir $input_run_inv
else
  echo "Directory is $input_run_inv already exist"
fi

export run_lcurves_output_dir=$OUTPUT_DIR/04_run_inv_lcurves
export run_lcurves_cov_output_dir=$OUTPUT_DIR/04_run_inv_lcurves_cov

if [ ! -d $run_lcurves_output_dir ]
then
  echo "Creating run directory $run_lcurves_output_dir"
  mkdir $run_lcurves_output_dir
else
  echo "Directory is $run_lcurves_output_dir already exist"
fi

if [ ! -d $run_lcurves_cov_output_dir ]
then
  echo "Creating run directory $run_lcurves_cov_output_dir"
  mkdir $run_lcurves_cov_output_dir
else
  echo "Directory is $run_lcurves_cov_output_dir already exist"
fi

toml set --toml-path smith_template.toml io.input_dir "$input_run_inv"
toml set --toml-path smith_template_cov.toml io.input_dir "$input_run_inv"

toml set --toml-path smith_template.toml io.output_dir "$run_lcurves_output_dir/output"
toml set --toml-path smith_template_cov.toml io.output_dir "$run_lcurves_cov_output_dir/output"

toml set --toml-path smith_template.toml io.diagnostics_dir "$run_lcurves_output_dir/diagnostics"
toml set --toml-path smith_template_cov.toml io.diagnostics_dir "$run_lcurves_cov_output_dir/diagnostics"
