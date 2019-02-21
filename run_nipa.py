import pandas as pd
from matplotlib import cm, pyplot as plt
import numpy as np
from climdiv_data import *
from simpleNIPA import *
from atmos_ocean_data import *
from utils import sstMap
import matplotlib as mpl
from utils import *
import pydap.client
import mpl_toolkits
import csv
import math


#### USER INPUT ####

# Select the input-output files:
index_file = './DATA/nao.txt'
clim_file = './DATA/APGD_prcpComo.txt'
filename = 'testComoNAO'

# Settings:
M = 2               # number of climate signal's phases
n_obs = 3           # number of observations (months)
lag = 3             # lag-time (months) --> 3 = seasonal
months = [1,2,3]    # months to consider (J,F,M)
startyr = 1971      # beginning of the time period to analyze
n_yrs = 38          # number of years to analyze

# Select the type of experiment:
# crv_flag:
#   True  = runs NIPA with crossvalidation
#   False = runs NIPA without crossvalidation and save the first SST-Principal Component for multi-variate model
#
crv_flag = True
map_flag = True

####################

# Don't mess with the next few lines
years = np.arange(startyr, startyr+n_yrs)

kwgroups = create_kwgroups(debug = True, climdata_months = months,
                        climdata_startyr = startyr, n_yrs = n_yrs,
                        n_mon_sst = n_obs, n_mon_index = n_obs, sst_lag = lag,
                        n_phases = M, phases_even = True,
                        index_lag = lag,
                        index_fp = index_file,
                        climdata_fp = clim_file)

climdata, sst, index, phaseind = get_data(kwgroups)
fp = './maps/%s' % (filename)
fig, axes = plt.subplots(M, 1, figsize = (6, 12))
timeseries = {'years' : [], 'data' : [], 'hindcast': []}
pc1 = {'pc1':[]}

print 'NIPA running...'

if M == 1:
    phase = 'allyears'
    model = NIPAphase(climdata, sst, index, phaseind[phase])
    model.phase = phase
    model.years = years[phaseind[phase]]
    model.bootcorr(corrconf = 0.95)
    model.gridCheck()
    model.crossvalpcr(xval = crv_flag)
    timeseries['years'] = model.years
    timeseries['data'] = model.clim_data
    timeseries['hindcast'] = model.hindcast
    print timeseries['years']
    print timeseries['data']
    
    if map_flag:
        fig, axes, m = sstMap(model, fig = fig, ax = axes)
        axes.set_title('%s, %.2f' % (phase, model.correlation))
        fig.savefig(fp)
        plt.close(fig)


else:
    for phase, ax in zip(phaseind, axes):
        model = NIPAphase(climdata, sst, index, phaseind[phase])
        model.phase = phase
        model.years = years[phaseind[phase]]
        model.bootcorr(corrconf = 0.95)
        model.gridCheck()
        model.crossvalpcr(xval = crv_flag)
        timeseries['years'].append(model.years)
        timeseries['data'].append(model.clim_data)
        timeseries['hindcast'].append(model.hindcast)
        if not crv_flag:
            if hasattr(model,'pc1'):
                pc1['pc1'].append(model.pc1)
        
        if map_flag:
            fig, ax, m = sstMap(model, fig = fig, ax = ax)
            ax.set_title('%s, %.2f' % (phase, model.correlation))
            fig.savefig(fp)
            plt.close(fig)


# save timeseries (exceptions handled only for 2 phase analysis)
if np.size(timeseries['hindcast'][0]) == 1:
    if math.isnan(timeseries['hindcast'][0]):
        # no result for the first phase -> consider only the second set of results
        timeseries['years'] = timeseries['years'][1]
        timeseries['data'] = timeseries['data'][1]
        timeseries['hindcast'] = timeseries['hindcast'][1]

elif np.size(timeseries['hindcast'][1]) == 1:
    if math.isnan(timeseries['hindcast'][1]):
        # no result for the second phase -> consider only the first set of results
        timeseries['years'] = timeseries['years'][0]
        timeseries['data'] = timeseries['data'][0]
        timeseries['hindcast'] = timeseries['hindcast'][0]

else:
    timeseries['years'] = np.concatenate(timeseries['years'])
    timeseries['data'] = np.concatenate(timeseries['data'])
    timeseries['hindcast'] = np.concatenate(timeseries['hindcast'])

df = pd.DataFrame(timeseries)
ts_file = './output/%s_timeseries.csv' % (filename)
df.to_csv(ts_file)

if not crv_flag:
    # save PC
    pc1['pc1'] = np.concatenate(pc1['pc1'])
    pc_file = './output/%s_pc1SST.csv' % (filename)
    df1 = pd.DataFrame(pc1)
    df1.to_csv(pc_file)

print 'NIPA run completed'







