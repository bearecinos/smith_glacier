mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/smith_template_ga_1e-04.toml |& tee log__ga_1e-04.txt
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/smith_template_ga_1e-03.toml |& tee log__ga_1e-03.txt
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/smith_template_ga_1e-02.toml |& tee log__ga_1e-02.txt
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/smith_template_ga_1e-01.toml |& tee log__ga_1e-01.txt
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/smith_template_ga_1e+00.toml |& tee log__ga_1e+00.txt
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/smith_template_ga_1e+01.toml |& tee log__ga_1e+01.txt
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/smith_template_ga_1e+02.toml |& tee log__ga_1e+02.txt
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/smith_template_ga_1e+03.toml |& tee log__ga_1e+03.txt
mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/smith_template_ga_1e+04.toml |& tee log__ga_1e+04.txt
OUT=$(tail "$RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/log__ga_1e+04.txt")
echo $OUT | mail -s "run_inv_lcurves on gamma_alpha finished" beatriz.recinos@ed.ac.uk
