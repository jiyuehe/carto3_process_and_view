# Copyright 2026 Jiyue He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%%
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import configuration

#%%
# setting
half_window_size = 400//2 # number of time points before and after the 2000 ms mark
directory = configuration.directory_setup()
name_prefix = configuration.map_name()

# load data
data = np.load(directory['data'] / f'{name_prefix}_carto.npz', allow_pickle=True)
carto = {k: data[k] for k in data.files}

data = np.load(directory['data'] / f'{name_prefix}_mesh.npz', allow_pickle=True)
mesh = {k: data[k] for k in data.files}

# grab the electrode positions and electrograms
electrode = carto['electrode']
electrode_positions_all = carto['electrode_positions']

electrode_positions = []
electrogram_unipolar_original = []
electrogram_bipolar_original = []
electrogram_reference_original = []
reference_channel_name = []
for e_id in range(len(electrode)):
    electrode_name = electrode[e_id]['unipolar_name']

    if 'mapping_' in electrode_name:
        # grab full-length electrograms
        uni = electrode[e_id]['unipolar']
        bi = electrode[e_id]['bipolar']

        if uni is not None and bi is not None:
            electrode_positions.append(electrode_positions_all[e_id, :])

            # ref = electrode[e_id]['reference']
            ref = electrode[e_id]['surface'][:,1] # surface lead V1

            electrogram_unipolar_original.append(uni)
            electrogram_bipolar_original.append(bi)
            electrogram_reference_original.append(ref)

            reference_channel_name.append(carto['electrode_point_info'][e_id]['reference_channel_name'])
            # reference_channel_name.append('V1')

electrode_positions = np.array(electrode_positions)
electrogram_unipolar_original = np.array(electrogram_unipolar_original)
electrogram_bipolar_original = np.array(electrogram_bipolar_original)
electrogram_reference_original = np.array(electrogram_reference_original)

debug_plot = 0
if debug_plot == 1:
    # plot electrograms of an electrode
    e_id = 350
    plt.figure(figsize=(12, 6))
    plt.plot(electrogram_reference_original[e_id, :], color = 'cyan', label='Reference Electrogram (original)')
    plt.plot(electrogram_unipolar_original[e_id, :], color = 'blue', label='Unipolar Electrogram (original)')
    plt.plot(electrogram_bipolar_original[e_id, :], color = 'magenta', label='Bipolar Electrogram (original)')
    plt.title('Original. Blue: unipolar, Magenta: bipolar, Cyan: reference')
    plt.xlabel('ms')
    plt.ylabel('mV')
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%
# mask the electrograms to the window of interest
t_start = 2000-1 - half_window_size # window of interest start time index
t_end = 2000-1 + half_window_size # window of interest end time index

taper_length = 50 # number of time points for gradual onset/offset at the window edges
taper_sigma = taper_length / 3 # sigma so the ramp reaches ~1% at the edge
taper = np.exp(-0.5 * ((np.arange(taper_length) - taper_length) / taper_sigma) ** 2) # Gaussian ramp from ~0 to 1
woi_window = np.zeros(electrogram_unipolar_original.shape[1])
woi_window[t_start:t_end] = 1.0
woi_window[t_start:t_start + taper_length] = taper
woi_window[t_end - taper_length:t_end] = taper[::-1]

electrogram_unipolar_masked = electrogram_unipolar_original * woi_window
electrogram_bipolar_masked = electrogram_bipolar_original * woi_window

# find QRS timing from the reference electrogram
s = np.abs(electrogram_reference_original)
s[:, :t_start] = 0
s[:, t_end:] = 0
qrs_time = [find_peaks(s[i, :], height=0.7*np.max(s[i, :]), distance=50)[0][0] for i in range(s.shape[0])] # find peaks in the derivative of each reference electrode

# mask out the QRS in the electrograms via inverse flat-top Gaussian window
qrs_taper_size = 50 # number of time points for the Gaussian ramp on each side
qrs_flat_size = 50  # number of time points held at zero in the flat middle region
qrs_taper_sigma = qrs_taper_size / 3 # sigma so the ramp reaches ~1% at the edge
n_electrodes = electrogram_unipolar_masked.shape[0]
n_sig = electrogram_unipolar_masked.shape[1]
qrs_taper_up = np.exp(-0.5 * ((np.arange(qrs_taper_size) - qrs_taper_size) / qrs_taper_sigma) ** 2) # ~0 -> 1
electrogram_unipolar = np.zeros_like(electrogram_unipolar_masked)
# electrogram_bipolar = np.zeros_like(electrogram_bipolar_masked)
for i in range(n_electrodes):
    qrs_bump = np.zeros(n_sig)
    qrs_start = qrs_time[i] - qrs_flat_size // 2 - qrs_taper_size
    qrs_flat_start = qrs_start + qrs_taper_size
    qrs_flat_end = qrs_flat_start + qrs_flat_size
    qrs_end = qrs_flat_end + qrs_taper_size
    qrs_bump[qrs_start:qrs_flat_start] = qrs_taper_up
    qrs_bump[qrs_flat_start:qrs_flat_end] = 1.0
    qrs_bump[qrs_flat_end:qrs_end] = qrs_taper_up[::-1]                                # 1 -> ~0
    qrs_window = 1 - qrs_bump # inverse: 1 outside QRS, flat zero at centre, smooth Gaussian tapers
    electrogram_unipolar[i, :] = electrogram_unipolar_masked[i, :] * qrs_window
    # electrogram_bipolar[i, :] = electrogram_bipolar_masked[i, :] * qrs_window
electrogram_bipolar = electrogram_bipolar_masked

debug_plot = 0
if debug_plot == 1:
    e_id = 350

    plt.figure(figsize=(12, 10))

    plt.subplot(5, 1, 1)
    plt.plot(electrogram_reference_original[e_id, :], color = 'cyan', label='Reference Electrogram (original)')
    plt.plot(electrogram_unipolar_original[e_id, :], color = 'blue', label='Unipolar Electrogram (original)')
    plt.plot(electrogram_bipolar_original[e_id, :], color = 'magenta', label='Bipolar Electrogram (original)')
    plt.title('Original. Blue: unipolar, Magenta: bipolar, Cyan: reference')
    plt.xlabel('ms')
    plt.ylabel('mV')

    plt.subplot(5, 1, 2)
    plt.plot(woi_window, color = 'blue')
    plt.title('Window of Interest')
    plt.xlabel('Time Points')
    plt.ylabel('Weight')

    plt.subplot(5, 1, 3)
    plt.plot(electrogram_unipolar_masked[e_id, :], color = 'blue', label='Unipolar Electro gram (masked)')
    plt.plot(electrogram_bipolar_masked[e_id, :], color = 'magenta', label='Bipolar Electrogram (masked)')
    plt.axvline(qrs_time[e_id], color='red', linestyle='--', label='QRS Timing')
    plt.title('Masked to window of interest. Blue: unipolar, Magenta: bipolar, Red dashed line: QRS timing')
    plt.xlabel('ms')
    plt.ylabel('mV')

    plt.subplot(5, 1, 4)
    plt.plot(qrs_window, color = 'blue')
    plt.title('Window for QRS Masking')
    plt.xlabel('Time Points')
    plt.ylabel('Weight')

    plt.subplot(5, 1, 5)
    plt.plot(electrogram_unipolar[e_id, :], color = 'blue', label='Unipolar Electrogram (masked)')
    plt.plot(electrogram_bipolar[e_id, :], color = 'magenta', label='Bipolar Electrogram (masked)')
    plt.title('QRS removed. Blue: unipolar, Magenta: bipolar')
    plt.xlabel('ms')
    plt.ylabel('mV')

    plt.tight_layout()
    plt.savefig(directory['result'] / f'{name_prefix}_QRS_removal.png', dpi=300) # save as png
    plt.close()

#%%
clinical_electrogram_unipolar_original = electrogram_unipolar_original
clinical_electrogram_bipolar_original = electrogram_bipolar_original
clinical_electrogram_unipolar_refined = electrogram_unipolar_original
clinical_electrogram_bipolar_refined = electrogram_bipolar_original
clinical_electrogram_reference = electrogram_reference_original

#%%
# sometimes the electrogram has high frequency noise such as 60 Hz noise from power supply etc, apply a moving average smoothing to remove them
window_size = 5 # number of time points in the moving average window
clinical_electrogram_unipolar_refined = np.convolve(clinical_electrogram_unipolar_refined.flatten(), np.ones(window_size)/window_size, mode='same').reshape(clinical_electrogram_unipolar_refined.shape)
clinical_electrogram_bipolar_refined = np.convolve(clinical_electrogram_bipolar_refined.flatten(), np.ones(window_size)/window_size, mode='same').reshape(clinical_electrogram_bipolar_refined.shape)

# activation time detection on bipolar electrogram
signal = clinical_electrogram_bipolar_refined
absolute_dvdt = np.abs(np.diff(signal, axis=1, prepend=signal[:, [0]]))
absolute_dvdt_woi = absolute_dvdt.copy()
for i in range(signal.shape[0]):
    absolute_dvdt_woi[i, :t_start] = 0
    absolute_dvdt_woi[i, t_end:] = 0

activation = np.zeros(absolute_dvdt_woi.shape[0], dtype=int)
for i in range(absolute_dvdt_woi.shape[0]):
    peaks, props = find_peaks(absolute_dvdt_woi[i, :], height=0.3*np.max(absolute_dvdt_woi[i, :]), distance=20)

    if len(peaks) == 0:
        activation[i] = 0
    elif len(peaks) == 1:
        activation[i] = peaks[0]
    else:
        heights = props['peak_heights']
        top2_order = np.argsort(heights)[-2:]  # indices into peaks/heights of 2 largest
        top2_peaks = peaks[top2_order]
        top2_heights = heights[top2_order]

        # sort descending by height
        desc = np.argsort(top2_heights)[::-1]
        top2_peaks = top2_peaks[desc]
        top2_heights = top2_heights[desc]

        # if 2nd largest is not too smaller than the largest, pick the earlier (smaller index) peak
        if top2_heights[1] >= 0.3 * top2_heights[0]:
            activation[i] = min(top2_peaks)
        else:
            activation[i] = top2_peaks[0]

debug_plot = 0
if debug_plot == 1:
    e_id = 350

    plt.figure(figsize=(12, 6))
    plt.plot(clinical_electrogram_bipolar_refined[e_id, :], color='blue', label='Bipolar Electrogram')
    plt.plot(absolute_dvdt_woi[e_id, :], color='orange', label='absolute dV/dt')
    plt.scatter(activation[e_id], absolute_dvdt_woi[e_id, activation[e_id]], color='red', label='Detected Activation Time')
    plt.title('Activation Time Detection from Bipolar Electrogram')
    plt.xlabel('ms')
    plt.ylabel('mV / mV/ms')
    plt.legend()
    plt.tight_layout()
    plt.show()

# remove activation time if it's the 1st or last point
for i in range(signal.shape[0]):
    if activation[i] == t_start or activation[i] == t_end:
        activation[i] = 0

# remove activation time if the signal is very small
do_flag = 1
if do_flag == 1:
    for e_id in range(signal.shape[0]):
        if np.max(clinical_electrogram_unipolar_refined[e_id, t_start:t_end]) - np.min(clinical_electrogram_unipolar_refined[e_id, t_start:t_end]) < 0.3: # > 1 mV is normal. < 0.5 mV is considered dense scar for unipolar
            activation[e_id] = 0
        if np.max(clinical_electrogram_bipolar_refined[e_id, t_start:t_end]) - np.min(clinical_electrogram_bipolar_refined[e_id, t_start:t_end]) < 0.2: # > 0.5 mV is normal. < 0.2 mV is considered dense scar for bipolar
            activation[e_id] = 0

#%% 
# save data
clinical_data = {}
for key, value in mesh.items():
    clinical_data[key] = value

clinical_data['electrode_positions'] = electrode_positions
clinical_data['clinical_electrogram_unipolar_original'] = clinical_electrogram_unipolar_original
clinical_data['clinical_electrogram_bipolar_original'] = clinical_electrogram_bipolar_original
clinical_data['clinical_electrogram_unipolar_refined'] = clinical_electrogram_unipolar_refined
clinical_data['clinical_electrogram_bipolar_refined'] = clinical_electrogram_bipolar_refined
clinical_data['clinical_electrogram_reference'] = clinical_electrogram_reference
clinical_data['clinical_electrogram_woi_start'] = t_start
clinical_data['clinical_electrogram_woi_end'] = t_end
clinical_data['clinical_activation_uni'] = activation
clinical_data['clinical_activation_bi'] = activation

file_path = directory['data'] / f'{name_prefix}_clinical.npz'
np.savez(file_path, **clinical_data)

#%%
import subprocess
import sys
subprocess.run([sys.executable, 'ui_patient_data_observer.py'], check=True)
