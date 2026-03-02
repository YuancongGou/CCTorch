import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
filedir = "./out_test"

misfit = np.load(filedir+"/misfit.npy")
phase = np.load(filedir+"/phase_grid.npy")
freq = np.load(filedir+"/freq.npy")

# normalized misfit at each frequency
nfreq = len(freq)
nphase = len(phase)
for ifreq in range(nfreq):
    maxv = np.max( np.abs( misfit[ifreq,:] ))
    misfit[ifreq,:] = misfit[ifreq,:] /maxv

# mk figure
fig = plt.figure()
ax = fig.add_subplot()
vmin = 0.5
vmax = 1
norm = colors.Normalize(vmin=vmin, vmax=vmax)
freqgrid,phasegrid = np.meshgrid(freq,phase)
bar = ax.pcolormesh(freqgrid,phasegrid,misfit.T,shading='nearest',cmap="viridis",norm=norm)

ax.set_xlabel("Frequancy [Hz]")
ax.set_ylabel("Phase velocity [km/sec]")

plt.colorbar(bar)
plt.savefig("out.png")