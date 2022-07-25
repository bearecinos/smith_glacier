import os
import numpy as np
import sys
from IPython import embed

print(sys.argv)

if (len(sys.argv)==1):
    print ('need arguments')
    exis()

if (len(sys.argv)==3):
    runs = np.load(sys.argv[1])
    script = sys.argv[2]
    if(np.size(runs)==0):
        print('we have done the last run');
        exit()
    else:
        print ('runs left:')
        print(runs)
        run1 = runs[0,:]
        if (np.shape(runs)[0]>1):
            runs = runs[1:,:]
        else:
            runs = []
        np.save('runs.npy',runs)
        ga = float(run1[0])
        gb = float(run1[1])
        da = float(run1[2])
        db = float(run1[3])
        std = int(run1[5])
        step = str(int(run1[4]))
else: 
    ga = float(sys.argv[1])
    gb = float(sys.argv[2])
    da = float(sys.argv[3])
    db = float(sys.argv[4])
    std = int(sys.argv[6])
    step = sys.argv[5]
    script = 'run_all_auto'

idoplot = False

base_toml = 'smith_cloud_subsampling'


alpha_pairs_L0_s = [(ga,da)]
#                    (6000.,100.), (1000.,100), (500.,100.)]
#               (12000.,100.)]

beta_pairs_L0_s =  [(gb,db)]



if (std==1):
 vel_file = 'smith_obs_vel_itslive-comp_std-adjusted-cloud_subsample-training-step-zero-' + step + 'E+0.h5' 
else:
 vel_file = 'smith_obs_vel_itslive-comp_itslive-cloud_subsample-training-step-zero-' + step + 'E+0_error-factor-1E+0.h5'


print(alpha_pairs_L0_s)
print(beta_pairs_L0_s)
print(vel_file)

from math import log10 , floor
def round_it(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1)

alpha_pairs_gamma_delt = []
beta_pairs_gamma_delt = []

for i in range(len(alpha_pairs_L0_s)):
    L0 = alpha_pairs_L0_s[i][0]
    s  = alpha_pairs_L0_s[i][1]
    delta_alpha = round_it(1./L0 * 1./s * 1./np.sqrt(4.*np.pi),2)
    gamma_alpha = round_it(delta_alpha * L0**2,2)
    alpha_pairs_gamma_delt = alpha_pairs_gamma_delt + [(gamma_alpha,delta_alpha)] 
    print('delta: ' + str(delta_alpha))
    print('gamma: ' + str(gamma_alpha))
    print('L0: ' + str(L0))
    print('s0: ' + str(s))

for i in range(len(beta_pairs_L0_s)):
    L0 = beta_pairs_L0_s[i][0]
    s  = beta_pairs_L0_s[i][1]
    delta = round_it(1./L0 * 1./s * 1./np.sqrt(4.*np.pi),2)
    gamma = round_it(delta * L0**2,2)
    beta_pairs_gamma_delt = beta_pairs_gamma_delt + [(gamma,delta)] 
    print('delta: ' + str(delta))
    print('gamma: ' + str(gamma))
    print('L0: ' + str(L0))
    print('s0: ' + str(s))

for i in range(len(alpha_pairs_gamma_delt)):

    gamma_alpha = alpha_pairs_gamma_delt[i][0]
    delta_alpha = alpha_pairs_gamma_delt[i][1]
    gamma_beta = beta_pairs_gamma_delt[0][0]
    delta_beta = beta_pairs_gamma_delt[0][1]
    print('delta: ' + str(delta_alpha))
    print('gamma: ' + str(gamma_alpha))

    new_toml = base_toml + 'step_' + str(step) + '_La_' + str(alpha_pairs_L0_s[i][0]) + '_Ca_' + str(alpha_pairs_L0_s[i][1]) \
                         + '_Lb_' + str(beta_pairs_L0_s[0][0]) + '_Cb_' + str(beta_pairs_L0_s[0][1]) + '_std_' + str(std)  + '.toml'
    base2_toml = base_toml + '.toml'

    submit_cmd = 'nohup bash ' + script + '.sh '  + str(24) + ' ' + str(step) + ' ' + \
                 vel_file + ' ' + base2_toml + ' ' + new_toml + ' ' + \
                 str(gamma_alpha) + ' ' + str(delta_alpha) + ' ' + \
                 str(gamma_beta) + ' ' + str(delta_beta) + ' ' + str(std) + ' &'
    print (submit_cmd)
    submit_cm2 = 'bash plot_all.sh ' + str(24) + ' ' + \
                 vel_file + ' ' + base2_toml + ' ' + new_toml + ' ' + \
                 str(gamma_alpha) + ' ' + str(delta_alpha) + ' ' + \
                 str(gamma_beta) + ' ' + str(delta_beta)
    if not idoplot:
     os.system(submit_cmd)
    else:
     os.system(submit_cm2)
