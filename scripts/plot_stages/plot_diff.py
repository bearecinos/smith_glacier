import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from scipy.stats import linregress

if (len(sys.argv)<2):
    raise Exception('Call with convergence .p file as argument')

if (len(sys.argv)==3):
    startind = int(sys.argv[2])
else:
    startind = 5000

f = open(sys.argv[1],'rb');

print(f)

q = pickle.load(f)

eignum = np.array(q[0])
sig = q[1]

ind = 0.5 * (eignum[1:]+eignum[0:-1])
diffs =  (np.diff(sig))/np.diff(eignum)

ind2 = ind[ind>startind]
diffs2 = diffs[ind>startind]

result = linregress(ind2,np.log(np.abs(diffs2)))
slope = result.slope
inter = result.intercept

print('slope is equal to ' + str(slope))
print('this means diff ~= exp(' + str(slope) + ' times k) where k is eig num')
print('this means that the full result could be smaller than the calculated one by about ' + str(np.abs(diffs[-1])/(1-np.exp(slope))))
print('but this is considered the extreme case')

plt.figure(1)
plt.plot(eignum,sig)
plt.grid()

plt.figure(2)
plt.semilogy(ind, np.abs(np.diff(sig))/np.diff(eignum))
plt.plot(ind2, np.exp(slope * ind2 + inter))
plt.text(8000,1e9,'slope=' + str(round(slope,5)) + '\n R^2=' + str(round(result.rvalue**2,3)))
plt.grid()

plt.savefig(os.path.join('/scratch/local/brecinos/smith_glacier/plots/.', 'sigma_change.png'), bbox_inches='tight', dpi=150)
