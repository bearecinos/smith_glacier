#!/bin/bash

tomls_main_dir=$RUN_CONFIG_DIR/run_workflow/itslive_exp_paper_tomls

echo $tomls_main_dir

tomls_main_dir_1=$tomls_main_dir/tomls_C0a2-1000_L0a-3000_C0b2-30_L0b-1000
tomls_main_dir_2=$tomls_main_dir/tomls_C0a2-500_L0a-3000_C0b2-60_L0b-1000
tomls_main_dir_3=$tomls_main_dir/tomls_C0a2-510_L0a-2000_C0b2-30_L0b-1000
tomls_main_dir_4=$tomls_main_dir/tomls_C0a2-150_L0a-3000_C0b2-30_L0b-1000

toml1=$tomls_main_dir_1/smith_itslive-std-adjusted-subsample-training-step-middle-8E+0_C0a2-1000_L0a-3000_C0b2-30_L0b-1000.toml
toml2=$tomls_main_dir_2/smith_itslive-std-adjusted-subsample-training-step-middle-8E+0_C0a2-500_L0a-3000_C0b2-60_L0b-1000.toml
toml3=$tomls_main_dir_3/smith_itslive-std-adjusted-subsample-training-step-middle-8E+0_C0a2-510_L0a-2000_C0b2-30_L0b-1000.toml
toml4=$tomls_main_dir_4/smith_itslive-std-adjusted-subsample-training-step-middle-8E+0_C0a2-150_L0a-3000_C0b2-30_L0b-1000.toml

#nohup bash $RUN_CONFIG_DIR/run_workflow/itslive_exp_paper_tomls/run_all_four.sh 24 $toml1 $toml2 $toml3 $toml4 >/dev/null 2>&1
