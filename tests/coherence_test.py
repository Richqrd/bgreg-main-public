from numpy import random, arange, sqrt, pi, sin, square, divide, multiply
import matplotlib.pyplot as plt
import matplotlib

from bgreg.data.analysis.fcn.metrics import coherence, butter, lfilter

matplotlib.use('Qt5Agg')

rng = random.default_rng()


fs = 10e3 #how many points per second we have (Hz) - i.e. 10kHz
N = 1e5 # how many points do we show in my signal on the graph
amp = 10
freq = 1000.0 #frequency that exists within the signal
noise_power = 0.001 * fs / 2 #introduce white noise
time = arange(N) / fs #how many time points will happen given sf and # of points
b, a = butter(2, 0.25, 'low') #filtering, lowpass - first param is order of filter which relates to resolution, second param is lowpass
sqrtnoise = sqrt(noise_power)
# x = rng.normal(scale=sqrt(noise_power), size=time.shape) #defines a signal using the normal distribution function, that includes this noise
# y = lfilter(b, a, x) #signal after filtering
temp = 2*pi*freq*time
x = sin(2*pi*freq*time) #create a signal, only 1 frequency, 1000Hz
# x += sin(2*pi*2000*time)
# y += rng.normal(scale=0.1*sqrt(noise_power), size=time.shape) #amplitude is scaled by 0.1
y = sin(2*pi*1000*time) #only 1 frequency, 2000Hz

# standard signal eqn = amplitude * sin (2pi * freq * time points considered)


# Calculate coherence
# nperseg = resolution in frequency representation
f, Cxy, ph, Pxx, Pyy, Pxy = coherence(x, y, fs, nperseg=512)

# Initialize plot values
plt.rcParams.update({'font.family': 'Arial'})
fig, axs = plt.subplots(6, figsize=(14, 14))
fontsize = 16
plt.subplots_adjust(hspace=0.7)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

fig.subplots_adjust(top=0.93)
# fig.suptitle('Spectral analysis for $s(x)$ and $s(y)$', fontsize=fontsize*1.3)
# other title
fig.suptitle('Spectral analysis for $s(x)$, where $P_{xx} = P_{yy}$', fontsize=fontsize*1.3)


# plot signals
ax_id = 0
axs[0].plot(x[:256], 'darkblue', marker='o', label='$s(x)$')
axs[0].plot(y[:256], 'r', linestyle='--', label='$s(y)$')
axs[ax_id].set_ylabel('Amplitude', fontsize=fontsize)
axs[ax_id].set_xlabel('Time points', fontsize=fontsize)
axs[ax_id].tick_params(labelsize=fontsize)
axs[ax_id].legend(fontsize=fontsize*.8, loc='upper right')


# plot autospectral densities
ax_id = 1
axs[ax_id].plot(f, Pxx.real, 'darkblue', label='$P_{xx}$')
axs[ax_id].plot(f, Pyy.real, 'r', linestyle='--', label='$P_{yy}$')
axs[ax_id].set_ylabel('PSD ($V^2/Hz$)', fontsize=fontsize)
axs[ax_id].set_xlabel('Frequency (Hz)', fontsize=fontsize)
axs[ax_id].tick_params(labelsize=fontsize)
axs[ax_id].legend(fontsize=fontsize*.8, loc='upper right')


# plot squared CSD
ax_id = 2
Pxy_squared = square(abs(Pxy))
axs[ax_id].plot(f, Pxy_squared.real, 'darkblue', label='$|P_{xy}|^2$')
axs[ax_id].set_ylabel('$|CSD|^2$', fontsize=fontsize)
axs[ax_id].set_xlabel('Frequency (Hz)', fontsize=fontsize)
axs[ax_id].tick_params(labelsize=fontsize)
axs[ax_id].legend(fontsize=fontsize*.8, loc='upper right')


# plot Pxx * Pyy
ax_id = 3
Pxx_divides_Pyy = divide(Pxx, Pyy)
Pxx_times_Pyy = multiply(Pxx, Pyy)
axs[ax_id].semilogy(f, Pxx_times_Pyy.real, 'darkblue', label=r'$P_{xx} \times P_{yy}$')
axs[ax_id].semilogy(f, Pxx_divides_Pyy.real, 'darkblue', linestyle='--', label=r'$P_{xx} \div P_{yy}$')
axs[ax_id].set_ylabel('PSD (log)', fontsize=fontsize)
axs[ax_id].set_xlabel('Frequency (Hz)', fontsize=fontsize)
axs[ax_id].tick_params(labelsize=fontsize)
axs[ax_id].legend(fontsize=fontsize*.8, loc='upper right')


# # Plot coherence
ax_id = 4
axs[ax_id].plot(f, Cxy.real, 'k', label='$C_{xy}$')
axs[ax_id].set_ylabel('Coherence', fontsize=fontsize)
axs[ax_id].set_xlabel('Frequency (Hz)', fontsize=fontsize)
axs[ax_id].tick_params(labelsize=fontsize)
axs[ax_id].legend(fontsize=fontsize*.8, loc='upper right')


# Plot phase
ax_id = 5
axs[ax_id].plot(f, ph, '+', color='k', label='phase x1 -> x2')
axs[ax_id].set_ylabel('Phase (deg)', fontsize=fontsize)
axs[ax_id].set_yticks([-180, -90, 0, 90, 180])
axs[ax_id].set_xlabel('Frequency (Hz)', fontsize=fontsize)
axs[ax_id].tick_params(labelsize=fontsize)


# visualizing and saving figure
# plt.savefig('spectral_s(x).svg', format='svg')
# other title
# plt.savefig('spectral_s(x)_s(y).svg', format='svg')
plt.show()
