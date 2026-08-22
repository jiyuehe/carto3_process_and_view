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
import configuration

#%%
# setting
directory = configuration.directory_setup()
name_prefix = configuration.map_name()

# load data
data = np.load(directory['data'] / f'{name_prefix}_carto.npz', allow_pickle=True)
carto = {k: data[k] for k in data.files}

#%%
# recording segments data
catheter = carto['catheter'].item()
mapping_name_unipolar = catheter['mapping_name_unipolar']
mapping_name_unipolar = [name.replace("MCC_Dx_UniPolar_", "") for name in mapping_name_unipolar]
surface_name = catheter['surface_name']

# some segments may have missing electrode positions, remove those segments from the analysis
segment_positions = catheter['mapping_position_unipolar']
segment_id_to_delete = []
for seg_idx, seg in enumerate(segment_positions):
    if seg is None:
        segment_id_to_delete.append(seg_idx)

catheter['mapping_electrogram_unipolar'] = [seg for idx, seg in enumerate(catheter['mapping_electrogram_unipolar']) if idx not in segment_id_to_delete]
catheter['surface_electrogram'] = [seg for idx, seg in enumerate(catheter['surface_electrogram']) if idx not in segment_id_to_delete]
catheter['mapping_position_unipolar'] = [seg for idx, seg in enumerate(catheter['mapping_position_unipolar']) if idx not in segment_id_to_delete]

mapping_electrogram_unipolar = catheter['mapping_electrogram_unipolar']
surface_electrogram = catheter['surface_electrogram']

# loop through each recording segment
n_segment = len(catheter['mapping_electrogram_unipolar']) # number of recording segments
mapping_electrogram_unipolar_activation = [None for _ in range(n_segment)]
mapping_electrogram_unipolar_qrs_subtracted = [None for _ in range(n_segment)]
for n in range(n_segment):
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
    # copy the global QRS peak indices for each unipolar channel
    qrs_peak_indices_per_channel = [np.copy(qrs_peak_indices) for _ in range(egm_unipolar.shape[1])]

    # adaptive qrs template length based on the far-field morphology of the unipolar electrograms
    signal = egm_unipolar
    peak_indices = qrs_peak_indices_per_channel
    qrs_half_window = utility.signal_processing.estimate_far_field_half_window(signal,peak_indices)
    template_len = 2 * qrs_half_window + 1

    half_window = qrs_half_window
    qrs_template_unipolar = utility.signal_processing.create_consistent_template(signal,peak_indices,half_window)

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

    mapping_electrogram_unipolar_qrs_subtracted[n] = qrs_subtracted

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
                dvdt_baseline + 2 * dvdt_noise,
                0.02 * np.max(finite_dvdt),
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

    # refine activation time detections
    # ------------------------------
    # adaptive qrs template length based on the far-field morphology of the unipolar electrograms
    signal = egm_unipolar
    peak_indices = activation_times_unipolar
    half_window = utility.signal_processing.estimate_far_field_half_window(signal,peak_indices)
    template_len = 2 * half_window + 1

    # for each unipolar electrogram, create morphology template from the detected activation times
    activation_template_unipolar = utility.signal_processing.create_consistent_template(signal,peak_indices,half_window,consistency_threshold=0.4)

    # for each unipolar electrogram channel, crosscorrelate the activation template with the QRS-subtracted unipolar electrogram to refine the activation time detections
    activation_times_unipolar_refined = [np.array([0], dtype=int) for _ in range(n_channels)]
    for channel_idx in range(n_channels):
        electrogram = qrs_subtracted[:, channel_idx].astype(float)
        template = activation_template_unipolar[channel_idx, :].astype(float)

        template_centered = template - np.median(template)
        electrogram_centered = electrogram - np.median(electrogram)

        # normalize the cross-correlation to find the lag that best aligns the template with this channel
        cross_correlation = np.correlate(electrogram_centered, template_centered, mode='same')

        # find out peak threshold
        correlation_baseline = np.median(cross_correlation)
        correlation_noise = 1.4826 * np.median(np.abs(cross_correlation - correlation_baseline))
        peak_height_threshold = max(
            correlation_baseline + 2 * correlation_noise,
            0.02 * np.max(cross_correlation),
        )
        peak_prominence_threshold = max(2 * correlation_noise, 0.0)

        correlation_peaks, _ = find_peaks(
            np.nan_to_num(cross_correlation, nan=-np.inf),
            distance=minimum_activation_distance,
            height=peak_height_threshold,
            prominence=peak_prominence_threshold,
        )

        # remove correlation peaks that are at the very beginning or very end of the signal, as they may be artifacts
        correlation_peaks = correlation_peaks[
            (correlation_peaks != 0) & (correlation_peaks != cross_correlation.shape[0])
        ]

        if len(correlation_peaks) != 0:
            activation_times_unipolar_refined[channel_idx] = correlation_peaks

    mapping_electrogram_unipolar_activation[n] = activation_times_unipolar_refined

    debug_plot = 0
    if debug_plot: # plot the original electrograms, activation template, and refined activation times
        sample_axis = np.arange(n_samples)
        trace_range = max(
            np.nanmax(np.ptp(egm_unipolar, axis=0)),
            np.nanmax(np.ptp(qrs_subtracted, axis=0)),
        )
        trace_spacing = max(1.0, trace_range * 1.05)
        trace_offsets = np.arange(n_channels) * trace_spacing

        template_axis_offset = np.arange(n_channels) * trace_spacing
        template_x = np.arange(-half_window, half_window + 1)
        surface_signal_scaled = surface_signal_sum * (trace_range / np.ptp(surface_signal_sum)) - trace_spacing

        fig, axes = plt.subplots(1, 3, figsize=(16, max(6, 0.25 * n_channels)), gridspec_kw={'width_ratios': [30, 1, 30]})
        original_axis, template_axis, subtracted_axis = axes

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
            activation_peaks = activation_times_unipolar_refined[channel_idx]
            if not np.array_equal(activation_peaks, [0]):
                subtracted_axis.scatter(
                    activation_peaks,
                    qrs_subtracted[activation_peaks, channel_idx] + offset,
                    color='red',
                    s=12,
                    zorder=3,
                )

            template = activation_template_unipolar[channel_idx, :]
            template_axis.plot(
                template_x,
                template + template_axis_offset[channel_idx],
                color='blue',
                linewidth=1.0,
            )

        surface_axis = np.arange(surface_signal_sum.shape[0])
        original_axis.plot(surface_axis, surface_signal_scaled, color='magenta', linewidth=1.0)
        subtracted_axis.plot(surface_axis, surface_signal_scaled, color='magenta', linewidth=1.0)

        original_axis.set_title('Original unipolar electrograms')
        template_axis.set_title('Activation template')
        subtracted_axis.set_title('QRS-subtracted electrograms and activation times')
        original_axis.set_xlabel('time (ms)')
        template_axis.set_xlabel('lag (samples)')
        subtracted_axis.set_xlabel('time (ms)')
        original_axis.set_yticks([])
        template_axis.set_yticks([])
        subtracted_axis.set_yticks([])
        original_axis.set_xlim(sample_axis[0], sample_axis[-1])
        template_axis.set_xlim(template_x[0], template_x[-1])

        # set the y-axis limits
        top_unipolar_egm = egm_unipolar[:, -1] + trace_offsets[-1]
        y_min = np.nanmin(surface_signal_scaled)
        y_max = np.nanmax(top_unipolar_egm)
        original_axis.set_ylim(y_min, y_max)
        subtracted_axis.set_ylim(y_min, y_max)
        template_axis.set_ylim(y_min, y_max)

        fig.tight_layout()

        fig_path = directory['result'] / f'activation_refined_{n}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()

#%%
# grab activations within a window of interest (WOI) around the 2000 ms mark for each recording segment
half_window_size_of_woi = 350//2 # number of time points before and after the 2000 ms mark
t_start = 2000-200 - half_window_size_of_woi # window of interest start time index
t_end = 2000-200 + half_window_size_of_woi # window of interest end time index

catheter['clinical_electrogram_woi_start'] = t_start
catheter['clinical_electrogram_woi_end'] = t_end

mapping_electrogram_unipolar_activation_within_woi = [None for _ in range(n_segment)]
for n in range(n_segment): # range(n_segment) or [some segment indices for debugging]
    print(f'recording segment id {n} in [0, {n_segment-1}]')

    activation_times_unipolar_refined = mapping_electrogram_unipolar_activation[n]
    if activation_times_unipolar_refined is None:
        activation_times_unipolar_refined = [np.array([], dtype=int) for _ in range(n_channels)]

    activation_times_within_woi = np.zeros((len(activation_times_unipolar_refined),), dtype=object)
    for channel_idx in range(len(activation_times_unipolar_refined)):
        activation_times = activation_times_unipolar_refined[channel_idx]
        if activation_times is not None:
            temp = activation_times[(activation_times >= t_start) & (activation_times <= t_end)]
            if len(temp) != 0:
                activation_times_within_woi[channel_idx] = temp[0]

    mapping_electrogram_unipolar_activation_within_woi[n] = activation_times_within_woi

catheter['mapping_electrogram_unipolar_activation'] = mapping_electrogram_unipolar_activation
catheter['mapping_electrogram_unipolar_activation_within_woi'] = mapping_electrogram_unipolar_activation_within_woi

catheter['mapping_electrogram_unipolar_qrs_subtracted'] = mapping_electrogram_unipolar_qrs_subtracted

# convert sequence-like values to object arrays to avoid heterogeneous-shape errors
catheter_to_save = {}
for k, v in catheter.items():
    # keep numpy arrays as-is, but coerce lists / sequences of arrays to object dtype
    if isinstance(v, np.ndarray):
        catheter_to_save[k] = v
    else:
        try:
            catheter_to_save[k] = np.asarray(v, dtype=object)
        except Exception:
            # fallback: store original value as a single-object array
            catheter_to_save[k] = np.asarray([v], dtype=object)

# allow_pickle=True ensures object arrays are saved correctly
file_path = directory['data'] / f'{name_prefix}_catheter.npz'
np.savez(file_path, **catheter_to_save, allow_pickle=True)

print('done')
