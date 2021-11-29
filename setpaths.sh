#!/bin/bash

export MAIN_PATH=/home/brecinos/scratch/smith_glacier
export INPUT_DIR=$MAIN_PATH/input_data
export PREPRO_STAGES=$MAIN_PATH/scripts/prepro_stages
export RUN_CONFIG_DIR=$MAIN_PATH/scripts/run_stages
export OUTPUT_DIR=$MAIN_PATH/output

# Paths to fenics_ice
export RUN_DIR=$FENICS_ICE_BASE_DIR/runs/


echo 'Input data directory is '$INPUT_DIR
echo 'Runs configuration directory is '$RUN_CONFIG_DIR
echo 'Prepro files run routines are in '$PREPRO_STAGES
echo 'All output directory is '$OUTPUT_DIR
echo 'Workflow run scripts '$RUN_DIR

