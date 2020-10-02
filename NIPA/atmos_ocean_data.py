#!/usr/bin/env python
"""
Module for loading atmospheric and oceanic data necessary to run NIPA
"""

import os
from os import environ as EV
import sys
import resource

def openDAPsst(version = '3b', debug = False, anomalies = True, **kwargs):
    """
    This function downloads data from the new ERSSTv3b on the IRI data library
    kwargs should contain: startyr, endyr, startmon, endmon, nbox
    """
    from utils import int_to_month
    from os.path import isfile
    from pydap.client import open_url
    from numpy import arange
    from numpy import squeeze
    import pickle
    import re
    from collections import namedtuple


    SSTurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version' + version + '/' + \
    '.anom/T/%28startmon%20startyr%29%28endmon%20endyr%29RANGEEDGES/T/nbox/0.0/boxAverage/dods'
    #SSTurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP-NCAR/.CDAS-1/.MONTHLY/.Intrinsic/.PressureLevel/.phi/P/%28700%29VALUES' +'/' + \
    #'.anom/T/%28startmon%20startyr%29%28endmon%20endyr%29RANGEEDGES/T/nbox/0.0/boxAverage/dods'
    #SSTurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version' + version + '/' + \
    #'.anom/T/%28startmon%20startyr%29%28endmon%20endyr%29RANGEEDGES/T/nbox/0.0/boxAverage/data.nc'

    if not anomalies:
       SSTurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version' + version + '/' + \
       '.sst/T/%28startmon%20startyr%29%28endmon%20endyr%29RANGEEDGES/T/nbox/0.0/boxAverage/dods'
        #SSTurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP-NCAR/.CDAS-1/.MONTHLY/.Intrinsic/.PressureLevel/.phi/P/%28700%29VALUES' +'/' + \
    #'.anom/T/%28startmon%20startyr%29%28endmon%20endyr%29RANGEEDGES/T/nbox/0.0/boxAverage/dods'

    print( 'Preparing to download from %s' % (SSTurl))

    i2m = int_to_month()

    # Keyword arguments for setting up the download
    DLargs = {
        'startmon'    : i2m[kwargs['months'][0]],
        'endmon'    : i2m[kwargs['months'][-1]],
        'startyr'    : str(kwargs['startyr']),
        'endyr'        : str(kwargs['endyr']),
        'nbox'         : str(kwargs['n_mon'])
            }
    fp = os.getcwd() + '/DATA/nipa/SST/' + DLargs['startmon'] + DLargs['startyr'] + \
        '_' + DLargs['endmon'] + DLargs['endyr'] + '_nbox_' + DLargs['nbox'] + '_version' + version

    fp = fp + '_anoms' if anomalies else fp + '_ssts'

    seasonal_var = namedtuple('seasonal_var', ('data','lat','lon'))

    #if isfile(fp):
        #print 'file found'
        #sys.setrecursionlimit(15000)
        #print sys.getrecursionlimit()
        # stupid edit: remove file
        #os.remove(fp)
        #if debug: print('Using pickled SST')
        #f = open(fp,'rb')
        #sstdata = pickle.load(f)
        #f.close()
        #var = seasonal_var(sstdata['grid'], sstdata['lat'], sstdata['lon'])
        #return var

    print( 'New SST field, will save to %s' % fp)
    print(SSTurl)
    for kw in DLargs:
        SSTurl = re.sub(kw, DLargs[kw], SSTurl)

    print('Starting download...')
    print(SSTurl)
    dataset = open_url(SSTurl)
    arg = 'anom' if anomalies else 'sst'
    sst = dataset[arg]

    time = dataset['T']
    grid = sst.array[:,:,:,:].data.squeeze() # MODIFIED ANDREA: inserted "data"
    t = time.data[:].squeeze()
    sstlat = dataset['Y'][:]
    sstlon = dataset['X'][:]
    print('Download finished.')

    #_Grid has shape (ntim, nlat, nlon)

    nseasons = 12 / kwargs['n_mon']
    if debug:
        print('Number of seasons is %i, number of months is %i' % (nseasons, kwargs['n_mon']))
    ntime = len(t)

    idx = arange(0, ntime, nseasons).astype(int)
    #print(idx)
    #print(grid)
    sst = grid[idx]
    sstdata = {'grid':sst, 'lat':sstlat, 'lon':sstlon}
    var = seasonal_var(sst, sstlat, sstlon)

    f = open(fp,'wb')
    pickle.dump(sstdata,f,pickle.HIGHEST_PROTOCOL)
    f.close()
    return var

def load_slp(newFormat = False, debug = False, anomalies = True, **kwargs):
    """
    This function loads HADSLP2r data.
    """
    from utils import slp_tf, int_to_month
    from netCDF4 import Dataset
    from sklearn.preprocessing import scale
    from numpy import arange, zeros, where
    from os.path import isfile
    import pandas as pd
    import pickle

    transform = slp_tf()    #This is for transforming kwargs into DLargs

    DLargs = {
        'startmon'    : transform[kwargs['months'][0]],
        'endmon'    : transform[kwargs['months'][-1]],
        'startyr'    : str(kwargs['startyr']),
        'endyr'        : str(kwargs['endyr']),
        'nbox'        : str(kwargs['n_mon'])
            }
    i2m = int_to_month() #_Use in naming convention
    fp = EV['DATA'] + '/nipa/SLP/' + i2m[kwargs['months'][0]] + \
        DLargs['startyr'] + '_' + i2m[kwargs['months'][-1]] + \
        DLargs['endyr'] + '_nbox_' + DLargs['nbox']

    if isfile(fp):
        f = open(fp)
        slpdata = pickle.load(f)
        f.close()
        if newFormat:
            from collections import namedtuple
            seasonal_var = namedtuple('seasonal_var', ('data','lat','lon'))
            slp = seasonal_var(slpdata['grid'], slpdata['lat'], slpdata['lon'])
            return slp
        return slpdata
    print('Creating new SLP pickle from netCDF file')

    #_Next block takes the netCDF file and extracts the time to make
    #_a time index.
    nc_fp = EV['DATA'] + '/netCDF/slp.mnmean.real.nc'
    dat = Dataset(nc_fp)
    t = dat.variables['time']
    extractargs = {
        'start'        : '1850-01',
        'periods'    : len(t[:]),
        'freq'        : 'M',
            }
    tiindexndex = pd.date_range(**extractargs)


    #Need to get start and end out of time index
    startyr = kwargs['startyr']
    startmon = int(DLargs['startmon'])

    idx_start = where((tiindexndex.year == startyr) & (tiindexndex.month == startmon))
    idx = []
    [idx.extend(arange(kwargs['n_mon']) + idx_start + 12*n) for n in range(kwargs['n_year'])]

    """
    This is how sst open dap does it but doesn't work for this
    idx = ((tiindexndex.year >= int(DLargs['startyr'])) & \
            ((tiindexndex.month >= int(DLargs['startmon'])) & \
             (tiindexndex.month <= int(DLargs['endmon'])))) & \
                ((tiindexndex.year <= int(DLargs['endyr'])))
    """


    if debug:
        print(tiindexndex[idx][:10])

    lat = dat.variables['lat'][:]
    lon = dat.variables['lon'][:]
    slp = dat.variables['slp'][:]

    nlat = len(lat)
    nlon = len(lon)
    time = tiindexndex[idx]
    slpavg = zeros((kwargs['n_year'], nlat, nlon))

    for year, mons in enumerate(idx):
        slpavg[year] = slp[mons].mean(axis=0)
        if debug:
            print('Averaging ', mons)

    #WHERE TO SCALE THE DATA?
    for i in range(nlat):
        for j in range(nlon):
            slpavg[:,i,j] = scale(slpavg[:,i,j])
    slpdata = {
            'grid'    :    slpavg,
            'lat'    :    lat,
            'lon'    :    lon
            }
    f = open(fp,'w')
    pickle.dump(slpdata,f)
    print('SLP data saved to %s' % (fp))
    f.close()
    if newFormat:
        from collections import namedtuple
        seasonal_var = namedtuple('seasonal_var', ('data','lat','lon'))
        slp = seasonal_var(slpdata['grid'], slpdata['lat'], slpdata['lon'])
        return slp
    return slpdata

def load_clim_file(fp, debug = False):
    # This function takes a specified input file, and
    # creates a pandas series with all necessary information
    # to run NIPA
    import numpy as np
    import pandas as pd

    #First get the description and years
    f = open(fp)
    description = f.readline()
    years = f.readline()
    startyr, endyr = years[:4], years[5:9]
    print( description)

    #First load extended index
    data = np.loadtxt(fp, skiprows = 2)
    nyrs = data.shape[0]
    data = data.reshape(data.size) # Make data 1D
    timeargs = {'start'     : startyr + '-01',
                'periods'    : len(data),
                'freq'        : 'M'}
    index = pd.date_range(**timeargs)
    clim_data = pd.Series(data = data, index = index)

    return clim_data

def load_climdata(**kwargs):
    
    data = load_clim_file(kwargs['fp'])
    from numpy import where, arange, zeros, inf
    from utils import slp_tf
    tran = slp_tf()
    startmon = int(tran[kwargs['months'][0]])
    startyr = kwargs['startyr']
    idx_start = where((data.index.year == startyr) & (data.index.month == startmon))
    idx = []
    [idx.extend(arange(len(kwargs['months'])) + idx_start + 12*n) for n in range(kwargs['n_year'])]
    #print len(idx)
    #print kwargs['n_year']
        #if kwargs['months'][-1]>12: # period extends across 2 years
        #del idx[-1]
        #print len(idx)
    climdata = zeros((kwargs['n_year']))
    for year, mons in enumerate(idx):
        climdata[year] = data.values[mons].mean()
    return climdata

def create_phase_index(debug = False, **kwargs):
    # kwargs = kwgroups['index']
    from numpy import sort
    index = load_clim_file(kwargs['fp'])
    from numpy import where, arange, zeros, inf
    from utils import slp_tf
    tran = slp_tf()
    startmon = int(tran[kwargs['months'][0]])
    startyr = kwargs['startyr']
    idx_start = where((index.index.year == startyr) & (index.index.month == startmon))
    idx = []
    [idx.extend(arange(kwargs['n_mon']) + idx_start + 12*n) for n in range(kwargs['n_year'])]
    index_avg = zeros((kwargs['n_year']))
    for year, mons in enumerate(idx):
        index_avg[year] = index.values[mons].mean()

    index = sort(index_avg)
    pos = index[index>0]
    neg = index[index<0]
    n_el = int(round(len(pos)*0.34))
    n_la = int(round(len(neg)*0.34))
    n_np = int(len(pos) - n_el)
    n_nn = int(len(neg) - n_la)


    cutoffs = {
        'la'    : (neg[0], neg[n_la-1]),
        'nn'    : (neg[n_la], neg[-1]),
        'np'    : (pos[0], pos[n_np-1]),
        'el'    : (pos[-n_el], pos[-1]),
        'N'        : (neg[n_la + 1], pos[n_np-1])
    }

    phaseind = {
            'pos'     : (index_avg >= cutoffs['el'][0]) & (index_avg <= \
                cutoffs['el'][1]),
            'neg'     : (index_avg >= cutoffs['la'][0]) & (index_avg <= \
                cutoffs['la'][1]),
            'neut'    : (index_avg >= cutoffs['N'][0]) & (index_avg <= \
                cutoffs['N'][1]),
            'neutpos'    : (index_avg >= cutoffs['np'][0]) & (index_avg <= \
                cutoffs['np'][1]),
            'neutneg'    : (index_avg >= cutoffs['nn'][0]) & (index_avg <= \
                cutoffs['nn'][1]),
            'allyears'    : (index_avg >= -inf)
            }


    return index_avg, phaseind

def create_phase_index2(**kwargs):
    from copy import deepcopy
    import numpy as np
    from numpy import sort
    index = load_clim_file(kwargs['fp'])
    from numpy import where, arange, zeros, inf
    from utils import slp_tf
    tran = slp_tf()
    startmon = int(tran[kwargs['months'][0]])
    startyr = kwargs['startyr']
    idx_start = where((index.index.year == startyr) & (index.index.month == startmon))
    idx = []
    [idx.extend(arange(kwargs['n_mon']) + idx_start + 12*n) for n in range(kwargs['n_year'])]
    index_avg = zeros((kwargs['n_year']))
    for year, mons in enumerate(idx):
        index_avg[year] = index.values[mons].mean()

    idx = np.argsort(index_avg)
    nyrs = kwargs['n_year']
    nphase = kwargs['n_phases']
    phases_even = kwargs['phases_even']
    p = np.zeros((len(index_avg)), dtype = 'bool')
    p1 = deepcopy(p)
    p2 = deepcopy(p)
    p3 = deepcopy(p)
    p4 = deepcopy(p)
    p5 = deepcopy(p)
    phaseind = {}
    if nphase == 1:
        p[idx[:]] = True
        phaseind['allyears'] = p
    if nphase == 2:
        x = nyrs / nphase
        p1[idx[:int(x)]] = True; phaseind['neg'] = p1
        p2[idx[int(x):]] = True; phaseind['pos'] = p2
    if nphase == 3:
        if phases_even:
            x = nyrs / nphase
            x2 = nyrs - x
        else:
            x = nphase / 4
            x2 = nyrs - x
        p1[idx[:x]] = True; phaseind['neg'] = p1
        p2[idx[x:x2]] = True; phaseind['neutral'] = p2
        p3[idx[x2:]] = True; phaseind['pos'] = p3

    if nphase == 4:
        if phases_even:
            x = nyrs / nphase
            x3 = nyrs - x
            xr = (x3 - x) / 2
            x2 = x+xr
        else:
            half = nyrs / 2
            x = int(round(half*0.34))
            x3 = nyrs - x
            xr = (x3 - x) / 2
            x2 = x + xr
        p1[idx[:x]] = True; phaseind['neg'] = p1
        p2[idx[x:x2]] = True; phaseind['neutneg'] = p2
        p3[idx[x2:x3]] = True; phaseind['netpos'] = p3
        p4[idx[x3:]] = True; phaseind['pos'] = p4
    if nphase == 5:
        if phases_even:
            x = nyrs / nphase
            x4 = nyrs - x
            xr = (x4 - x) / 3
            x2 = x+xr
            x3 = x4-xr
        else:
            half = nyrs / 2
            x = int(round(half*0.3))
            x4 = nyrs - x
            xr = (x4 - x) / 3
            x2 = x+xr
            x3 = x4-xr
        p1[idx[:x]] = True; phaseind['neg'] = p1
        p2[idx[x:x2]] = True; phaseind['neutneg'] = p2
        p3[idx[x2:x3]] = True; phaseind['neutral'] = p3
        p4[idx[x3:x4]] = True; phaseind['neutpos'] = p4
        p5[idx[x4:]] = True; phaseind['pos'] = p5
    # if nphase == 6:
    return index_avg, phaseind
