import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import interpolate 

ds_its_2014 = xr.open_dataset('its_live_antarctica/ANT_G0240_2014.nc');
ds_its_2018 = xr.open_dataset('its_live_antarctica/ANT_G0240_2018.nc');
ds_meas = xr.open_dataset('measures/measures_2000_2020/Antarctica_ice_velocity_2013_2014_1km_v01.nc')

x_its = ds_its_2014['x'].values
y_its = np.flipud(ds_its_2014['y'].values)

vx_its_2014 = np.flipud(ds_its_2014['vx'].values)
vy_its_2014 = np.flipud(ds_its_2014['vy'].values)
vx_its_2014 = vx_its_2014[5750:7250,4000:6000]
vy_its_2014 = vy_its_2014[5750:7250,4000:6000]

vxerr_its_2014 = np.flipud(ds_its_2014['vx_err'].values * np.sqrt(ds_its_2014['count'].values))
vyerr_its_2014 = np.flipud(ds_its_2014['vy_err'].values * np.sqrt(ds_its_2014['count'].values))
vxerr_its_2014 = vxerr_its_2014[5750:7250,4000:6000]
vyerr_its_2014 = vyerr_its_2014[5750:7250,4000:6000]


x_its = x_its[4000:6000]
y_its = y_its[5750:7250]

vx_its_2018 = np.flipud(ds_its_2018['vx'].values)
vy_its_2018 = np.flipud(ds_its_2018['vy'].values)
vx_its_2018 = vx_its_2018[5750:7250,4000:6000]
vy_its_2018 = vy_its_2018[5750:7250,4000:6000]

x_meas = ds_meas['x'].values[1150:1500]
y_meas = np.flipud(ds_meas['y'].values)[2000:2400]

vx_meas = np.flipud(ds_meas['VX'].values)[2000:2400,1150:1500]
vy_meas = np.flipud(ds_meas['VY'].values)[2000:2400,1150:1500]
vxerr_meas = np.flipud(ds_meas['STDX'].values)[2000:2400,1150:1500]
vyerr_meas = np.flipud(ds_meas['STDY'].values)[2000:2400,1150:1500]


x_meas_grid, y_meas_grid = np.meshgrid(x_meas,y_meas)
x_its_grid, y_its_grid = np.meshgrid(x_its,y_its)



f_interp_vx = interpolate.LinearNDInterpolator(list(zip(x_meas_grid.ravel(),y_meas_grid.ravel())),vx_meas.ravel())
f_interp_vy = interpolate.LinearNDInterpolator(list(zip(x_meas_grid.ravel(),y_meas_grid.ravel())),vy_meas.ravel())

f_interp_vxerr = interpolate.LinearNDInterpolator(list(zip(x_meas_grid.ravel(),y_meas_grid.ravel())),vxerr_meas.ravel())
f_interp_vyerr = interpolate.LinearNDInterpolator(list(zip(x_meas_grid.ravel(),y_meas_grid.ravel())),vyerr_meas.ravel())

vx_meas_interp = f_interp_vx(x_its_grid,y_its_grid)
vy_meas_interp = f_interp_vy(x_its_grid,y_its_grid)

vxerr_meas_interp = f_interp_vxerr(x_its_grid,y_its_grid)
vyerr_meas_interp = f_interp_vyerr(x_its_grid,y_its_grid)

plt.figure()
plt.contourf(np.sqrt((vx_its_2018-vx_its_2014)**2+(vy_its_2018-vy_its_2014)**2)/8.,np.linspace(0,100,20)); 
plt.colorbar(); 

plt.figure()
plt.contourf(np.sqrt((vx_its_2014-vx_meas_interp)**2+(vy_its_2014-vy_meas_interp)**2),np.linspace(0,100,20)); 
plt.colorbar(); 

ratio_itslive = np.sqrt((vxerr_its_2014)**2+(vyerr_its_2014)**2) / \
                np.sqrt((vx_its_2014-vx_meas_interp)**2+(vy_its_2014-vy_meas_interp)**2)

plt.figure()
plt.contourf(ratio_itslive,np.linspace(0,5,21)); 
plt.colorbar(); 

plt.figure()
plt.contourf(np.sqrt((vxerr_its_2014)**2+(vyerr_its_2014)**2),np.linspace(0,100,20)); 
plt.colorbar(); 



plt.show()











