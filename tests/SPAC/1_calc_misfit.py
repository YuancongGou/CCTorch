import numpy as np
import os
import sub_spac_das
#----------------------------------------
# parameter
#----------------------------------------
# savedir
savedir = "./out_test"
# filedir
filedir = "./data_cross-spectra"
# mk phase velocity grid (km/s)
phase_search = np.arange(0.1,4,0.01)

#----------------------------------------
# main 
#----------------------------------------
os.makedirs(savedir,exist_ok=True)
freq = np.load(f"{filedir}/freq.npy")
cross_real = np.load(f"{filedir}/cross_real.npy") # real-part of cross-spectra
distlist = np.load(f"{filedir}/distance.npy") # intersation distance

nfreq = len(freq)
nccr = len(distlist)
nphase = len(phase_search)

misfit = np.zeros([nfreq,nphase])
run = sub_spac_das.fit_spac_strain()
for ifreq in range(nfreq):
    cross_obs = cross_real[:,ifreq]
    f = freq[ifreq]
    
    cost_list,Amp_search,syn_spac_search = run.grid_search_fitting_rayleigh(f,cross_obs,distlist,phase_search)
    misfit[ifreq,:] = cost_list[:]

np.save(savedir+"/misfit",misfit)
np.save(savedir+"/freq",freq)
np.save(savedir+"/phase_grid",phase_search)