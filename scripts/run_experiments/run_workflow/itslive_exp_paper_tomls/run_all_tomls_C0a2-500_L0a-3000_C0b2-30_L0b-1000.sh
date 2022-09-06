#!/bin/bash

tomls_main_dir=$RUN_CONFIG_DIR/run_workflow/itslive_exp_paper_tomls/tomls_C0a2-500_L0a-3000_C0b2-30_L0b-1000

echo $tomls_main_dir

toml1=$tomls_main_dir/smith_itslive-std-adjusted-subsample-training-step-middle-8E+0_C0a2-500_L0a-3000_C0b2-30_L0b-1000.toml
toml2=$tomls_main_dir/smith_itslive-std-adjusted-complete_C0a2-500_L0a-3000_C0b2-30_L0b-1000.toml
toml3=$tomls_main_dir/smith_itslive-std-adjusted-subsample-training-step-middle-4E+0_C0a2-500_L0a-3000_C0b2-30_L0b-1000.toml
toml4=$tomls_main_dir/smith_itslive-subsample-training-step-middle-8E+0_C0a2-500_L0a-3000_C0b2-30_L0b-1000.toml

#nohup bash $RUN_CONFIG_DIR/run_workflow/itslive_exp_paper_tomls/run_all_four.sh 24 $toml1 $toml2 $toml3 $toml4 >/dev/null 2>&1
