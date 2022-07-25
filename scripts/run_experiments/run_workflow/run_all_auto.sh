#!/bin/bash

#'sbatch run_all.slurm ' + str(24) + ' ' +  step\
#                 vel_file + ' ' + base2_toml + ' ' + new_toml + ' ' + \
#                 str(gamma_alpha) + ' ' + str(delta_alpha) + ' ' + \
#                 str(gamma_beta) + ' ' + str(delta_beta)


echo $#
echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7
echo $8
echo $9
echo ${10}


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
cp $MAIN_PATH/output/01_mesh/smith_variable_ocean* $input_run_inv/.
# Gridded data files
cp $MAIN_PATH/output/02_gridded_data/smith_bedmachine.h5 $input_run_inv/.
cp $MAIN_PATH/output/02_gridded_data/smith_smb.h5 $input_run_inv/.
cp $MAIN_PATH/output/02_gridded_data/smith_bglen.h5 $input_run_inv/.
cp $MAIN_PATH/output/02_gridded_data/smith_obs_vel_* $input_run_inv/.
cp $MAIN_PATH/output/02_gridded_data/smith_melt_depth_params.h5 $input_run_inv/.

# Create output directory for inversion output
#export run_inv_output_dir=$OUTPUT_DIR/03_run_inv
export run_inv_output_dir=$OUTPUT_DIR/03_step_$2_ga_$6_da_$7_gb_$8_db_$9_std_${10}
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

echo $run_inv_output_dir

base_toml=$RUN_CONFIG_DIR/run_workflow/itslive_exp_tomls/$4
new_toml=$RUN_CONFIG_DIR/run_workflow/itslive_exp_tomls/$5

echo "copying tomls $5"
cp $base_toml $new_toml



toml set --toml-path $new_toml io.input_dir "$input_run_inv"
toml set --toml-path $new_toml io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $new_toml io.diagnostics_dir "$run_inv_output_dir/diagnostics"
toml set --toml-path $new_toml --to-float inversion.gamma_alpha $6
toml set --toml-path $new_toml --to-float inversion.delta_alpha $7
toml set --toml-path $new_toml --to-float inversion.gamma_beta $8
toml set --toml-path $new_toml --to-float inversion.delta_beta $9
toml set --toml-path $new_toml --to-float inversion.delta_beta_gnd $9
toml set --toml-path $new_toml obs.vel_file $3


dirsubname=$(echo $5 | cut -c24- | rev | cut -c6- | rev);
echo "GOT HERE 2"
echo $dirsubname

plotting_dir=exp_$dirsubname
CONFIGFILE=$MAIN_PATH/config.ini
TOMLFILE=$new_toml

echo $(date -u) "Run started"
echo $(date -u) "run started" | mail -s "run started" dngoldberg@gmail.com

var="true"
toml set --toml-path $new_toml --to-bool io.write_diagnostics $var
toml set --toml-path $new_toml io.log_level "info"
mpirun -n $1 ./unmute.sh 0 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $new_toml  > $run_inv_output_dir/out.inv 2> $run_inv_output_dir/err.inv
#mpirun -n $1 ./unmute.sh 0 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $new_toml  
var="false"
toml set --toml-path $new_toml --to-bool io.write_diagnostics $var
toml set --toml-path $new_toml io.log_level "error"

cd $MAIN_PATH/scripts/plot_stages
python plot_inversion.py -conf $CONFIGFILE -toml $TOMLFILE -sub_plot_dir=$plotting_dir
cd $OLDPWD

echo $(date -u) "Done with Inversion!"
echo $(date -u) "done with inversion" | mail -s "inversion" dngoldberg@gmail.com

python run_all.py

exit 0

mpirun -n $1 ./unmute.sh 0 python $FENICS_ICE_BASE_DIR/runs/run_forward.py $new_toml > $run_inv_output_dir/out.fwd 2> $run_inv_output_dir/err.fwd

cd $MAIN_PATH/scripts/plot_stages
python plot_dq_ts.py -conf $CONFIGFILE -toml $TOMLFILE -sub_plot_dir=$plotting_dir -n_sens 3 14
cd $OLDPWD

echo $(date -u) "Done with Forward!"
echo $(date -u) "done with fwd" | mail -s "forward" dngoldberg@gmail.com

mpirun -n $1 ./unmute.sh 0 python $FENICS_ICE_BASE_DIR/runs/run_eigendec.py $new_toml > $run_inv_output_dir/out.eig 2> $run_inv_output_dir/err.eig

cd $MAIN_PATH/scripts/plot_stages
python plot_eigen_info.py -conf $CONFIGFILE -toml $TOMLFILE -sub_plot_dir=$plotting_dir -n_eigen 1 2 200 4000
cd $OLDPWD

echo $(date -u) "Done with eigendec -------------------------------------------------------!"
echo $(date -u) "done with eigendec" | mail -s "eigendec" dngoldberg@gmail.com

mpirun -n $1 ./unmute.sh 0 python $FENICS_ICE_BASE_DIR/runs/run_errorprop.py $new_toml > $run_inv_output_dir/out.err 2> $run_inv_output_dir/err.err

cd $MAIN_PATH/scripts/plot_stages
python plot_path.py -conf $CONFIGFILE -toml $TOMLFILE -sub_plot_dir=$plotting_dir
cd $OLDPWD

echo $(date -u) "Done with error propagation"
echo $(date -u) "done with errorprop" | mail -s "errorpro" dngoldberg@gmail.com

mpirun -n $1 ./unmute.sh 0 python $FENICS_ICE_BASE_DIR/runs/run_invsigma.py $new_toml > $run_inv_output_dir/out.inv 2> $run_inv_output_dir/err.inv

echo $(date -u) "We are done with the whole workflow!"
echo $(date -u) "done with workflow" | mail -s "end" dngoldberg@gmail.com

