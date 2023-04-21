mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/smith_template_da_1e-07.toml |& tee log__da_1e-07
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/smith_template_da_1e-06.toml |& tee log__da_1e-06
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/smith_template_da_1e-05.toml |& tee log__da_1e-05
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/smith_template_da_1e-04.toml |& tee log__da_1e-04
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/smith_template_da_1e-03.toml |& tee log__da_1e-03
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/smith_template_da_1e-02.toml |& tee log__da_1e-02
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/smith_template_da_1e-01.toml |& tee log__da_1e-01
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/smith_template_da_1e+00.toml |& tee log__da_1e+00
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/smith_template_da_1e+01.toml |& tee log__da_1e+01
OUT=$(tail "$RUN_CONFIG_DIR/run_lcurves_paper/tomls_delta_alpha/log__da_1e+01.txt")
echo $OUT | mail -s "run_inv_lcurves on delta_alpha finished" beatriz.recinos@ed.ac.uk
