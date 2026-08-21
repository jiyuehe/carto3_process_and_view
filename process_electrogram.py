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
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import utility

#%%
# setting
half_window_size_of_woi = 300//2 # number of time points before and after the 2000 ms mark
directory = utility.configuration.directory_setup()
name_prefix = utility.configuration.map_name()

# load data
data = np.load(directory['data'] / f'{name_prefix}_carto.npz', allow_pickle=True)
carto = {k: data[k] for k in data.files}

data = np.load(directory['data'] / f'{name_prefix}_mesh.npz', allow_pickle=True)
mesh = {k: data[k] for k in data.files}

#%%
# recording segments data
catheter = carto['catheter'].item()
mapping_position_unipolar = catheter['mapping_position_unipolar']
mapping_electrogram_unipolar = catheter['mapping_electrogram_unipolar']
surface_electrogram = catheter['surface_electrogram']
mapping_name_unipolar = catheter['mapping_name_unipolar']
mapping_name_unipolar = [name.replace("MCC_Dx_UniPolar_", "") for name in mapping_name_unipolar]
surface_name = catheter['surface_name']

# loop through each recording segment
n_segment = len(catheter['mapping_position_unipolar']) # number of recording segments
for n in [0]: #range(n_segment):
    print(f'recording segment id {n} in [0, {n_segment-1}]')

    egm_unipolar = mapping_electrogram_unipolar[n]
    egm_surface = surface_electrogram[n]

    # QRS timing detection from the surface ECG
    # ------------------------------
    # combine the 12 surface ECG leads into one signal for QRS detection
    surface_signal = np.abs(egm_surface) # take abs first so positive and negative deflections do not cancel each other
    surface_signal_sum = np.sum(surface_signal, axis=1)

    # smooth to reduce noise and jitter
    smooth_window = 30
    smooth_kernel = np.ones(smooth_window) / smooth_window
    surface_signal_smooth = np.convolve(surface_signal_sum, smooth_kernel, mode='same')

    # find the QRS timings
    min_qrs_spacing = 300 # ms
    qrs_half_window = 50 # ms before/after each detected QRS peak
    qrs_threshold = np.percentile(surface_signal_smooth, 95)
    qrs_peak_indices, _ = find_peaks(surface_signal_smooth, distance=min_qrs_spacing, height=qrs_threshold)

    debug_plot = 0
    if debug_plot: # show the surface ECG leads and detected QRS peaks
        fig, ax = plt.subplots(figsize=(12, 8))
        n_channel = egm_surface.shape[1]

        # determine spacing based on signal magnitudes to separate traces visually
        per_channel_magnitude = np.ptp(egm_surface, axis=0) # peak-to-peak voltage
        max_magnitude = np.nanmax(per_channel_magnitude) 
        spacing = max_magnitude * 1.5
        offsets = np.arange(n_channel) * spacing

        # plot each channel with an offset
        for channel_idx in range(n_channel):
            ax.plot(egm_surface[:, channel_idx] + offsets[channel_idx], color='blue', linewidth=1)

        # plot the summed signal scaled and placed at bottom
        surface_signal_sum_scaled = surface_signal_sum * (max_magnitude / np.ptp(surface_signal_sum)) - spacing
        surface_signal_smooth_scaled = surface_signal_smooth * (max_magnitude / np.ptp(surface_signal_smooth)) - spacing

        ax.plot(surface_signal_sum_scaled, color='magenta', linewidth=1)
        ax.plot(surface_signal_smooth_scaled, color='blue', linewidth=1)
        ax.scatter(qrs_peak_indices, surface_signal_sum_scaled[qrs_peak_indices], color='red', s=20, zorder=5)
        
        ax.set_xlabel('Time (ms)')
        yticks = np.concatenate(([-spacing], offsets))
        ylabels = ['sum'] + list(surface_name)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_title(f'Surface ECGs and QRS detection (segment id {n} of [0, {n_segment-1}])')
        plt.tight_layout()

    # create QRS morphology template for each of the unipolar electrograms
    # ------------------------------
    qrs_half_window = utility.signal_processing.estimate_far_field_half_window(egm_unipolar,qrs_peak_indices) # adaptive qrs template length based on the far-field morphology of the unipolar electrograms
    template_len = 2 * qrs_half_window + 1

    qrs_template_unipolar, _ = utility.signal_processing.create_consistent_template(egm_unipolar,qrs_peak_indices,qrs_half_window)

    # subtract the QRS template from each unipolar electrogram to remove the QRS component
    # ------------------------------
    qrs_subtracted = egm_unipolar.copy()
    qrs_taper_size = 15
    qrs_subtraction_window = np.ones(template_len, dtype=float)
    qrs_taper = 0.5 * (1 - np.cos(np.pi * np.arange(qrs_taper_size) / qrs_taper_size))
    qrs_subtraction_window[:qrs_taper_size] = qrs_taper
    qrs_subtraction_window[-qrs_taper_size:] = qrs_taper[::-1]

    for channel_idx in range(egm_unipolar.shape[1]):
        for peak_idx in qrs_peak_indices:
            start = peak_idx - qrs_half_window
            end = peak_idx + qrs_half_window + 1
            seg_start = max(0, start)
            seg_end = min(egm_unipolar.shape[0], end)

            signal_segment = egm_unipolar[seg_start:seg_end, channel_idx]

            if peak_idx <= qrs_half_window:
                missing_front = qrs_half_window - peak_idx
                template_segment = qrs_template_unipolar[channel_idx, missing_front:missing_front + signal_segment.shape[0]]
            elif peak_idx + qrs_half_window + 1 > egm_unipolar.shape[0]:
                template_segment = qrs_template_unipolar[channel_idx, :signal_segment.shape[0]]
            else:
                template_segment = qrs_template_unipolar[channel_idx, :]

            if peak_idx <= qrs_half_window:
                subtraction_window = qrs_subtraction_window[missing_front:missing_front + signal_segment.shape[0]]
            elif peak_idx + qrs_half_window + 1 > egm_unipolar.shape[0]:
                subtraction_window = qrs_subtraction_window[:signal_segment.shape[0]]
            else:
                subtraction_window = qrs_subtraction_window

            # only subtract when needed
            template_magnitude = np.ptp(template_segment)
            signal_magnitude = np.ptp(signal_segment)
            if template_magnitude > 0.2 and signal_magnitude > 0.2:  # only subtract if both template and signal have significant magnitude
                qrs_subtracted[seg_start:seg_end, channel_idx] -= template_segment * subtraction_window # the subtraction is tapered to avoid abrupt changes by multiplying with a tapering window (subtraction_window)

    debug_plot = 0
    if debug_plot: # plot the original, QRS template, and QRS-subtracted electrograms
        n_channels = egm_unipolar.shape[1]
        template_x = np.arange(-qrs_half_window, qrs_half_window + 1) # x-axis for the QRS template

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(1, 3, width_ratios=[20, 1, 20], wspace=0.08)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_mid = fig.add_subplot(gs[0, 1])
        ax_right = fig.add_subplot(gs[0, 2])

        y_spacing = 2.0
        y_offset = np.arange(n_channels) * y_spacing

        if len(qrs_peak_indices) > 0:
            for peak_idx in qrs_peak_indices:
                ax_left.axvline(peak_idx, color='red', linewidth=1.0, clip_on=True) # the vertical line indicating QRS timing

        for channel_idx in range(n_channels):
            signal_trace = egm_unipolar[:, channel_idx] + y_offset[channel_idx]
            ax_left.plot(signal_trace, color='blue', linewidth=1.0)

        for channel_idx in range(qrs_template_unipolar.shape[0]):
            template = qrs_template_unipolar[channel_idx, :]
            ax_mid.plot(template_x, template + channel_idx * y_spacing, color='blue', linewidth=1.2)

        if len(qrs_peak_indices) > 0:
            for peak_idx in qrs_peak_indices:
                ax_right.axvline(peak_idx, color='red', linewidth=1.0, clip_on=True) # the vertical line indicating QRS timing

        for channel_idx in range(qrs_subtracted.shape[1]):
            signal_trace = qrs_subtracted[:, channel_idx] + y_offset[channel_idx]
            ax_right.plot(signal_trace, color='blue', linewidth=1.0)

        # find out the global min and max across all three subplots to set a shared y-axis range
        stacked_trace_values = []
        for channel_idx in range(n_channels):
            stacked_trace_values.append(egm_unipolar[:, channel_idx] + y_offset[channel_idx])
            stacked_trace_values.append(qrs_subtracted[:, channel_idx] + y_offset[channel_idx])
            stacked_trace_values.append(qrs_template_unipolar[channel_idx, :] + channel_idx * y_spacing)
        all_min = min(np.min(v) for v in stacked_trace_values)
        all_max = max(np.max(v) for v in stacked_trace_values)
        shared_y_margin = 1.0
        for ax in [ax_left, ax_mid, ax_right]:
            ax.set_ylim(all_min - shared_y_margin, all_max + shared_y_margin)

        ax_left.set_title('Original EGMs')
        ax_left.set_yticks([])
        ax_mid.set_title('QRS templates')
        ax_mid.set_yticks([])
        ax_right.set_title('QRS-subtracted EGMs')
        ax_right.set_yticks([])

        for ax in [ax_left, ax_mid, ax_right]:
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig_path = directory['result'] / f'qrs_subtraction_{n}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()

    # activation time detection on the QRS-subtracted unipolar electrograms
    # ------------------------------
    # for each qrs subtractedunipolar electrogram, do auto correlation to find out its cycle length
    n_samples, n_channels = qrs_subtracted.shape
    cycle_length_unipolar = np.full(n_channels, np.nan, dtype=float)
    autocorrelations = []
    for channel_idx in range(n_channels):
        electrogram = qrs_subtracted[:, channel_idx].astype(float)
        electrogram = electrogram - np.mean(electrogram)

        autocorrelation = np.correlate(electrogram, electrogram, mode='full')[n_samples - 1:]
        autocorrelation /= autocorrelation[0]
        autocorrelations.append(autocorrelation)
        candidate_lags, properties = find_peaks(autocorrelation,prominence=0.05)

        if len(candidate_lags) > 0:
            candidate_heights = autocorrelation[candidate_lags]
            # finite_candidates = np.isfinite(candidate_heights)
            # candidate_lags = candidate_lags[finite_candidates]
            # candidate_heights = candidate_heights[finite_candidates]
            # if len(candidate_lags) == 0:
            #     continue

            minimum_candidate_height = 0.3 * np.max(candidate_heights)
            fundamental_candidates = candidate_lags[candidate_heights >= minimum_candidate_height]
            if len(fundamental_candidates) == 0:
                fundamental_candidates = np.array([candidate_lags[np.argmax(candidate_heights)]])
            cycle_length_unipolar[channel_idx] = fundamental_candidates[0]

    # the cycle length of this recording segment is the median of the cycle lengths of all unipolar electrograms
    finite_cycle_lengths = cycle_length_unipolar[np.isfinite(cycle_length_unipolar)]
    cycle_lengths = np.median(finite_cycle_lengths) if len(finite_cycle_lengths) > 0 else np.nan

    debug_plot = 0
    if debug_plot: # plot the QRS-subtracted unipolar electrograms and their autocorrelations
        fig, axes = plt.subplots(n_channels, 2, figsize=(14, 4 * n_channels), squeeze=False)
        sample_axis = np.arange(n_samples)
        lag_axis = np.arange(n_samples)

        for channel_idx in range(n_channels):
            signal_axis = axes[channel_idx, 0]
            autocorrelation_axis = axes[channel_idx, 1]

            signal_axis.plot(sample_axis, qrs_subtracted[:, channel_idx], color='blue')
            signal_axis.set_title(f'QRS-subtracted unipolar signal (channel {channel_idx})')

            autocorrelation = autocorrelations[channel_idx]
            autocorrelation_axis.plot(lag_axis, autocorrelation, color='purple')
            if np.isfinite(cycle_length_unipolar[channel_idx]):
                cycle_length = int(cycle_length_unipolar[channel_idx])
                autocorrelation_axis.scatter(
                    cycle_length,
                    autocorrelation[cycle_length],
                    color='red',
                )
            autocorrelation_axis.set_title(f'Autocorrelation (channel {channel_idx})')
            autocorrelation_axis.set_xlabel('Lag (ms)')
            autocorrelation_axis.set_ylabel('Normalized autocorrelation')

        fig.tight_layout()
        plt.show()

    # for each qrs subtracted unipolar electrogram, find out the activation time by finding the peak of the maximum negative slope (dv/dt)
    minimum_activation_distance = int(np.floor(0.8 * cycle_lengths))

    activation_times_unipolar = []
    for channel_idx in range(n_channels):
        electrogram = qrs_subtracted[:, channel_idx].astype(float)
        negative_dvdt = -np.diff(electrogram, prepend=electrogram[0])

        if np.any(np.isfinite(negative_dvdt)):
            finite_dvdt = negative_dvdt[np.isfinite(negative_dvdt)]
            dvdt_baseline = np.median(finite_dvdt)
            dvdt_noise = 1.4826 * np.median(np.abs(finite_dvdt - dvdt_baseline))
            peak_height_threshold = max(
                dvdt_baseline + 4 * dvdt_noise,
                0.05 * np.max(finite_dvdt),
            )
            peak_prominence_threshold = max(2 * dvdt_noise, 0.0)
            activation_peaks, _ = find_peaks(
                np.nan_to_num(negative_dvdt, nan=-np.inf),
                distance=minimum_activation_distance,
                height=peak_height_threshold,
                prominence=peak_prominence_threshold,
            )
        else:
            activation_peaks = np.array([], dtype=int)

        # remove activation peaks that are at the very beginning or very end of the signal, as they may be artifacts
        activation_peaks = activation_peaks[(activation_peaks != 0) & (activation_peaks != qrs_subtracted.shape[0])]

        activation_times_unipolar.append(activation_peaks)

    debug_plot = 0
    if debug_plot: # plot the before and after QRS-subtracted unipolar electrograms and the detected activation times
        sample_axis = np.arange(n_samples)
        trace_range = max(
            np.nanmax(np.ptp(egm_unipolar, axis=0)),
            np.nanmax(np.ptp(qrs_subtracted, axis=0)),
        )
        trace_spacing = max(1.0, trace_range * 1.2)
        trace_offsets = np.arange(n_channels) * trace_spacing

        surface_signal_scaled = surface_signal_sum * (trace_range / np.ptp(surface_signal_sum)) - trace_spacing

        fig, axes = plt.subplots(1, 2, figsize=(12, max(6, 0.25 * n_channels)), sharex=True)
        original_axis, subtracted_axis = axes

        for channel_idx in range(n_channels):
            offset = trace_offsets[channel_idx]
            original_axis.plot(
                sample_axis,
                egm_unipolar[:, channel_idx] + offset,
                color='blue',
                linewidth=0.7,
            )
            subtracted_axis.plot(
                sample_axis,
                qrs_subtracted[:, channel_idx] + offset,
                color='blue',
                linewidth=0.7,
            )
            activation_peaks = activation_times_unipolar[channel_idx]
            subtracted_axis.scatter(
                activation_peaks,
                qrs_subtracted[activation_peaks, channel_idx] + offset,
                color='red',
                s=12,
                zorder=3,
            )

        surface_axis = np.arange(surface_signal_sum.shape[0])
        original_axis.plot(surface_axis, surface_signal_scaled, color='magenta', linewidth=1.0)
        subtracted_axis.plot(surface_axis, surface_signal_scaled, color='magenta', linewidth=1.0)

        original_axis.set_title('Original unipolar electrograms')
        subtracted_axis.set_title('QRS-subtracted electrograms and activation times')
        original_axis.set_xlabel('time (ms)')
        subtracted_axis.set_xlabel('time (ms)')
        original_axis.set_yticks([])
        subtracted_axis.set_yticks([])
        original_axis.set_xlim(sample_axis[0], sample_axis[-1])

        # set the y-axis limits
        top_unipolar_egm = egm_unipolar[:, -1] + trace_offsets[-1]
        y_min = np.nanmin(surface_signal_scaled)
        y_max = np.nanmax(top_unipolar_egm)
        original_axis.set_ylim(y_min, y_max)
        subtracted_axis.set_ylim(y_min, y_max)

        fig.tight_layout()
        plt.show()

        # fig_path = directory['result'] / f'activation_{n}.png'
        # plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        # plt.close()

#%%
    # refine activation time detections
    # ------------------------------
    # for each unipolar electrogram, create morphology template from the detected activation times
    qrs_half_window = utility.signal_processing.estimate_far_field_half_window(egm_unipolar,activation_times_unipolar) # adaptive qrs template length based on the far-field morphology of the unipolar electrograms
    template_len = 2 * qrs_half_window + 1

    qrs_template_unipolar, _ = utility.signal_processing.create_consistent_template(egm_unipolar,activation_times_unipolar,qrs_half_window)


#%%





# # grab the electrode positions and electrograms
# electrode = carto['electrode']
# electrode_positions_all = carto['electrode_positions']

# electrode_positions = []
# electrogram_unipolar_original = []
# electrogram_bipolar_original = []
# electrogram_reference_original = []
# reference_channel_name = []
# for e_id in range(len(electrode)):
#     electrode_name = electrode[e_id]['unipolar_name']

#     if 'mapping_' in electrode_name:
#         # grab full-length electrograms
#         uni = electrode[e_id]['unipolar']
#         bi = electrode[e_id]['bipolar']

#         if uni is not None and bi is not None:
#             electrode_positions.append(electrode_positions_all[e_id, :])

#             # ref = electrode[e_id]['reference']
#             ref = electrode[e_id]['surface'][:,1] # surface lead V1

#             electrogram_unipolar_original.append(uni)
#             electrogram_bipolar_original.append(bi)
#             electrogram_reference_original.append(ref)

#             reference_channel_name.append(carto['electrode_point_info'][e_id]['reference_channel_name'])
#             # reference_channel_name.append('V1')

# electrode_positions = np.array(electrode_positions)
# electrogram_unipolar_original = np.array(electrogram_unipolar_original)
# electrogram_bipolar_original = np.array(electrogram_bipolar_original)
# electrogram_reference_original = np.array(electrogram_reference_original)

# debug_plot = 0
# if debug_plot == 1:
#     # plot electrograms of an electrode
#     e_id = 350
#     plt.figure(figsize=(12, 6))
#     plt.plot(electrogram_reference_original[e_id, :], color = 'cyan', label='Reference Electrogram (original)')
#     plt.plot(electrogram_unipolar_original[e_id, :], color = 'blue', label='Unipolar Electrogram (original)')
#     plt.plot(electrogram_bipolar_original[e_id, :], color = 'magenta', label='Bipolar Electrogram (original)')
#     plt.title('Original. Blue: unipolar, Magenta: bipolar, Cyan: reference')
#     plt.xlabel('ms')
#     plt.ylabel('mV')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# #%%
# # mask the electrograms to the window of interest
# t_start = 2000-1 - half_window_size_of_woi # window of interest start time index
# t_end = 2000-1 + half_window_size_of_woi # window of interest end time index

# taper_length = 50 # number of time points for gradual onset/offset at the window edges
# taper_sigma = taper_length / 3 # sigma so the ramp reaches ~1% at the edge
# taper = np.exp(-0.5 * ((np.arange(taper_length) - taper_length) / taper_sigma) ** 2) # Gaussian ramp from ~0 to 1
# woi_window = np.zeros(electrogram_unipolar_original.shape[1])
# woi_window[t_start:t_end] = 1.0
# woi_window[t_start:t_start + taper_length] = taper
# woi_window[t_end - taper_length:t_end] = taper[::-1]

# electrogram_unipolar_masked = electrogram_unipolar_original * woi_window
# electrogram_bipolar_masked = electrogram_bipolar_original * woi_window

# # find QRS timing from the reference electrogram
# s = np.abs(electrogram_reference_original)
# s[:, :t_start] = 0
# s[:, t_end:] = 0
# qrs_time = [find_peaks(s[i, :], height=0.7*np.max(s[i, :]), distance=50)[0][0] for i in range(s.shape[0])] # find peaks in the derivative of each reference electrode

# # mask out the QRS in the electrograms via inverse flat-top Gaussian window
# qrs_taper_size = 50 # number of time points for the Gaussian ramp on each side
# qrs_flat_size = 50  # number of time points held at zero in the flat middle region
# qrs_taper_sigma = qrs_taper_size / 3 # sigma so the ramp reaches ~1% at the edge
# n_electrodes = electrogram_unipolar_masked.shape[0]
# n_sig = electrogram_unipolar_masked.shape[1]
# qrs_taper_up = np.exp(-0.5 * ((np.arange(qrs_taper_size) - qrs_taper_size) / qrs_taper_sigma) ** 2) # ~0 -> 1
# electrogram_unipolar = np.zeros_like(electrogram_unipolar_masked)
# # electrogram_bipolar = np.zeros_like(electrogram_bipolar_masked)
# for i in range(n_electrodes):
#     qrs_bump = np.zeros(n_sig)
#     qrs_start = qrs_time[i] - qrs_flat_size // 2 - qrs_taper_size
#     qrs_flat_start = qrs_start + qrs_taper_size
#     qrs_flat_end = qrs_flat_start + qrs_flat_size
#     qrs_end = qrs_flat_end + qrs_taper_size
#     qrs_bump[qrs_start:qrs_flat_start] = qrs_taper_up
#     qrs_bump[qrs_flat_start:qrs_flat_end] = 1.0
#     qrs_bump[qrs_flat_end:qrs_end] = qrs_taper_up[::-1]                                # 1 -> ~0
#     qrs_window = 1 - qrs_bump # inverse: 1 outside QRS, flat zero at centre, smooth Gaussian tapers
#     electrogram_unipolar[i, :] = electrogram_unipolar_masked[i, :] * qrs_window
#     # electrogram_bipolar[i, :] = electrogram_bipolar_masked[i, :] * qrs_window
# electrogram_bipolar = electrogram_bipolar_masked

# debug_plot = 0
# if debug_plot == 1:
#     e_id = 350

#     plt.figure(figsize=(12, 10))

#     plt.subplot(5, 1, 1)
#     plt.plot(electrogram_reference_original[e_id, :], color = 'cyan', label='Reference Electrogram (original)')
#     plt.plot(electrogram_unipolar_original[e_id, :], color = 'blue', label='Unipolar Electrogram (original)')
#     plt.plot(electrogram_bipolar_original[e_id, :], color = 'magenta', label='Bipolar Electrogram (original)')
#     plt.title('Original. Blue: unipolar, Magenta: bipolar, Cyan: reference')
#     plt.xlabel('ms')
#     plt.ylabel('mV')

#     plt.subplot(5, 1, 2)
#     plt.plot(woi_window, color = 'blue')
#     plt.title('Window of Interest')
#     plt.xlabel('Time Points')
#     plt.ylabel('Weight')

#     plt.subplot(5, 1, 3)
#     plt.plot(electrogram_unipolar_masked[e_id, :], color = 'blue', label='Unipolar Electro gram (masked)')
#     plt.plot(electrogram_bipolar_masked[e_id, :], color = 'magenta', label='Bipolar Electrogram (masked)')
#     plt.axvline(qrs_time[e_id], color='red', linestyle='--', label='QRS Timing')
#     plt.title('Masked to window of interest. Blue: unipolar, Magenta: bipolar, Red dashed line: QRS timing')
#     plt.xlabel('ms')
#     plt.ylabel('mV')

#     plt.subplot(5, 1, 4)
#     plt.plot(qrs_window, color = 'blue')
#     plt.title('Window for QRS Masking')
#     plt.xlabel('Time Points')
#     plt.ylabel('Weight')

#     plt.subplot(5, 1, 5)
#     plt.plot(electrogram_unipolar[e_id, :], color = 'blue', label='Unipolar Electrogram (masked)')
#     plt.plot(electrogram_bipolar[e_id, :], color = 'magenta', label='Bipolar Electrogram (masked)')
#     plt.title('QRS removed. Blue: unipolar, Magenta: bipolar')
#     plt.xlabel('ms')
#     plt.ylabel('mV')

#     plt.tight_layout()
#     plt.savefig(directory['result'] / f'{name_prefix}_QRS_removal.png', dpi=300) # save as png
#     plt.close()

# #%%
# clinical_electrogram_unipolar_original = electrogram_unipolar_original
# clinical_electrogram_bipolar_original = electrogram_bipolar_original
# clinical_electrogram_unipolar_refined = electrogram_unipolar_original
# clinical_electrogram_bipolar_refined = electrogram_bipolar_original
# clinical_electrogram_reference = electrogram_reference_original

# #%%
# # sometimes the electrogram has high frequency noise such as 60 Hz noise from power supply etc, apply a moving average smoothing to remove them
# window_size = 5 # number of time points in the moving average window
# clinical_electrogram_unipolar_refined = np.convolve(clinical_electrogram_unipolar_refined.flatten(), np.ones(window_size)/window_size, mode='same').reshape(clinical_electrogram_unipolar_refined.shape)
# clinical_electrogram_bipolar_refined = np.convolve(clinical_electrogram_bipolar_refined.flatten(), np.ones(window_size)/window_size, mode='same').reshape(clinical_electrogram_bipolar_refined.shape)

# # activation time detection on bipolar electrogram
# signal = clinical_electrogram_bipolar_refined
# absolute_dvdt = np.abs(np.diff(signal, axis=1, prepend=signal[:, [0]]))
# absolute_dvdt_woi = absolute_dvdt.copy()
# for i in range(signal.shape[0]):
#     absolute_dvdt_woi[i, :t_start] = 0
#     absolute_dvdt_woi[i, t_end:] = 0

# activation = np.zeros(absolute_dvdt_woi.shape[0], dtype=int)
# for i in range(absolute_dvdt_woi.shape[0]):
#     peaks, props = find_peaks(absolute_dvdt_woi[i, :], height=0.3*np.max(absolute_dvdt_woi[i, :]), distance=20)

#     if len(peaks) == 0:
#         activation[i] = 0
#     elif len(peaks) == 1:
#         activation[i] = peaks[0]
#     else:
#         heights = props['peak_heights']
#         top2_order = np.argsort(heights)[-2:]  # indices into peaks/heights of 2 largest
#         top2_peaks = peaks[top2_order]
#         top2_heights = heights[top2_order]

#         # sort descending by height
#         desc = np.argsort(top2_heights)[::-1]
#         top2_peaks = top2_peaks[desc]
#         top2_heights = top2_heights[desc]

#         # if 2nd largest is not too smaller than the largest, pick the earlier (smaller index) peak
#         if top2_heights[1] >= 0.3 * top2_heights[0]:
#             activation[i] = min(top2_peaks)
#         else:
#             activation[i] = top2_peaks[0]

# debug_plot = 0
# if debug_plot == 1:
#     e_id = 350

#     plt.figure(figsize=(12, 6))
#     plt.plot(clinical_electrogram_bipolar_refined[e_id, :], color='blue', label='Bipolar Electrogram')
#     plt.plot(absolute_dvdt_woi[e_id, :], color='orange', label='absolute dV/dt')
#     plt.scatter(activation[e_id], absolute_dvdt_woi[e_id, activation[e_id]], color='red', label='Detected Activation Time')
#     plt.title('Activation Time Detection from Bipolar Electrogram')
#     plt.xlabel('ms')
#     plt.ylabel('mV / mV/ms')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # remove activation time if it's the 1st or last point
# for i in range(signal.shape[0]):
#     if activation[i] == t_start or activation[i] == t_end:
#         activation[i] = 0

# # remove activation time if the signal is very small
# do_flag = 1
# if do_flag == 1:
#     for e_id in range(signal.shape[0]):
#         if np.max(clinical_electrogram_unipolar_refined[e_id, t_start:t_end]) - np.min(clinical_electrogram_unipolar_refined[e_id, t_start:t_end]) < 0.3: # > 1 mV is normal. < 0.5 mV is considered dense scar for unipolar
#             activation[e_id] = 0
#         if np.max(clinical_electrogram_bipolar_refined[e_id, t_start:t_end]) - np.min(clinical_electrogram_bipolar_refined[e_id, t_start:t_end]) < 0.2: # > 0.5 mV is normal. < 0.2 mV is considered dense scar for bipolar
#             activation[e_id] = 0

# #%% 
# # save data
# clinical_data = {}
# for key, value in mesh.items():
#     clinical_data[key] = value

# clinical_data['electrode_positions'] = electrode_positions
# clinical_data['clinical_electrogram_unipolar_original'] = clinical_electrogram_unipolar_original
# clinical_data['clinical_electrogram_bipolar_original'] = clinical_electrogram_bipolar_original
# clinical_data['clinical_electrogram_unipolar_refined'] = clinical_electrogram_unipolar_refined
# clinical_data['clinical_electrogram_bipolar_refined'] = clinical_electrogram_bipolar_refined
# clinical_data['clinical_electrogram_reference'] = clinical_electrogram_reference
# clinical_data['clinical_electrogram_woi_start'] = t_start
# clinical_data['clinical_electrogram_woi_end'] = t_end
# clinical_data['clinical_activation_uni'] = activation
# clinical_data['clinical_activation_bi'] = activation

# file_path = directory['data'] / f'{name_prefix}_clinical.npz'
# np.savez(file_path, **clinical_data)

print('done')
