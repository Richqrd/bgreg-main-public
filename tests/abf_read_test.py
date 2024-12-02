import os

from bgreg.data.io import get_raw_signal_abf
from bgreg.native.datapaths import datapath

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


filename = os.path.join(datapath, *["cfc-test", "2018_11_08_0022.abf"])
raw_abf = get_raw_signal_abf(filename)

# Initialize plot values
plt.rcParams.update({'font.family': 'Arial'})
fig, axs = plt.subplots(2, figsize=(10, 10))
fontsize = 22
plt.subplots_adjust(hspace=0.7)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# plot signals
ax_id = 0
axs[ax_id].plot(raw_abf.data[0], 'darkblue')
axs[ax_id].set_ylabel('Voltage (mV)', fontsize=fontsize)
axs[ax_id].set_xlabel('Time points', fontsize=fontsize)
axs[ax_id].set_ylim([-1, 1])
axs[ax_id].tick_params(labelsize=fontsize)
axs[ax_id].set_title('Hippocampal Recording', size=fontsize, fontweight='bold')

# plot autospectral densities
ax_id = 1
axs[ax_id].plot(raw_abf.data[1], 'red')
axs[ax_id].set_ylabel('Voltage (mV)', fontsize=fontsize)
axs[ax_id].set_xlabel('Time points', fontsize=fontsize)
axs[ax_id].tick_params(labelsize=fontsize)
axs[ax_id].set_title('Contralateral Thalamic Recording', size=fontsize, fontweight='bold')
axs[ax_id].set_ylim([-1, 2])

# Uncomment to see plot (commented out to simplify pytest)
# plt.show()
