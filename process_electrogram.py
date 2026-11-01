# Copyright 2026 Jiyue He
# Contact: jiyuehe@alumni.upenn.edu
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
from pathlib import Path
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import utility

import configuration
import sys
sys.path.insert(0, str(configuration.COMMON_DIR))
import common

#%%
directory = configuration.directory_setup()
patients = common.read_info_csv(directory)

n_left_atrium = 0
n_loaded = 0
for i in range(0,len(patients)):
    if patients[i]['atrium_flag'] == 'LA':
        n_left_atrium = n_left_atrium + 1

    if patients[i]['status'] == 'loaded':
        n_loaded = n_loaded + 1

print(f'Total number of left atrium clinical maps: {n_left_atrium}. Loaded {n_loaded} maps.')

#%%
suffix = '_carto_catheter.npz'
for map_id in range(0,len(patients)):
    patient = patients[map_id]

    if patient['atrium_flag'] == 'LA':
        name_prefix = f'{patient['id']}_{patient['mesh']}'
        print(f'process {name_prefix}')

        # load electrogram recording data
        # ------------------------------------------------------------
        data = np.load(directory['data'] / f'{name_prefix}{suffix}', allow_pickle=True)
        catheter = {k: data[k] for k in data.files}

        # recording segments data
        mapping_name_unipolar = catheter['mapping_name_unipolar']
        surface_name = catheter['surface_name']

        # some segments may have missing electrode positions, remove those segments from the analysis
        segment_positions = catheter['mapping_position_unipolar']
        segment_id_to_delete = []
        for seg_idx, seg in enumerate(segment_positions):
            if seg is None: # it is missing all electrode positions
                segment_id_to_delete.append(seg_idx)

        segment_ids_to_delete = set(segment_id_to_delete)
        original_segment_count = len(segment_positions)
        for field_name, field_values in catheter.items():
            if isinstance(field_values, list) and len(field_values) == original_segment_count:
                catheter[field_name] = [
                    value
                    for segment_id, value in enumerate(field_values)
                    if segment_id not in segment_ids_to_delete
                ]

        # plot the mapping catheter and surface lead electrograms
        debug_plot = 0
        if debug_plot: 
            segment_id = 10
            result_folder = directory['result']
            mapping_electrogram_unipolar = catheter['mapping_electrogram_unipolar']
            mapping_name_unipolar = catheter['mapping_name_unipolar']
            surface_electrogram = catheter['surface_electrogram']
            surface_name = catheter['surface_name']
            utility.debug_plots.mapping_and_surface_electrogram(segment_id, result_folder, mapping_name_unipolar, mapping_electrogram_unipolar, surface_name, surface_electrogram)

        # loop through each recording segment
        # ------------------------------------------------------------
        mapping_electrogram_unipolar = catheter['mapping_electrogram_unipolar']
        surface_electrogram = catheter['surface_electrogram']
        n_segment = len(catheter['mapping_electrogram_unipolar']) # number of recording segments
        mapping_electrogram_unipolar_activation = [None for _ in range(n_segment)]
        mapping_electrogram_unipolar_qrs_subtracted = [None for _ in range(n_segment)]
        surface_ecg_sum = [None for _ in range(n_segment)]
        cycle_lengths_per_segment = [None for _ in range(n_segment)]
        for n in range(n_segment):
            if n % (n_segment//4) == 0:
                print(f'    recording segment id {n} in [0, {n_segment-1}]')

            # QRS timing detection from the surface ECG
            # ------------------------------
            # combine the 12 surface ECG leads into one signal for QRS detection
            egm_surface = surface_electrogram[n]

            sampling_rate_hz = 1000
            surface_bandpass_hz = (10, 400)
            surface_filter = butter(4, surface_bandpass_hz, btype='bandpass', fs=sampling_rate_hz, output='sos')
            egm_surface_filtered = sosfiltfilt(surface_filter, egm_surface, axis=0)
            
            surface_signal = np.abs(egm_surface_filtered) # take abs first so positive and negative deflections do not cancel each other
            surface_signal_sum = np.sum(surface_signal, axis=1)

            # smooth to reduce noise and jitter
            smooth_window = 30
            smooth_kernel = np.ones(smooth_window) / smooth_window
            surface_signal_smooth = np.convolve(surface_signal_sum, smooth_kernel, mode='same')

            surface_ecg_sum[n] = surface_signal_smooth / np.ptp(surface_signal_smooth)

            # find the QRS timings
            min_qrs_spacing = 300 # ms
            qrs_half_window = 50 # ms before/after each detected QRS peak
            qrs_threshold = np.percentile(surface_signal_smooth, 95)
            qrs_peak_indices, _ = find_peaks(surface_signal_smooth, distance = min_qrs_spacing, height = qrs_threshold)

            # show the surface ECGs and detected QRS peaks
            debug_plot = 0
            if debug_plot: 
                result_folder = directory['result']
                surface_name = catheter['surface_name']
                utility.debug_plots.qrs_timings_on_surface_electrograms(n, result_folder, surface_name, egm_surface, egm_surface_filtered, surface_signal_sum, surface_signal_smooth, qrs_peak_indices)

            # bandpass filter the unipolar electrogram
            # ------------------------------
            egm_unipolar = np.asarray(mapping_electrogram_unipolar[n], dtype=float)
            unipolar_bandpass_hz = (10, 400)
            unipolar_filter = butter(4, unipolar_bandpass_hz, btype='bandpass', fs=sampling_rate_hz, output='sos')
            egm_unipolar_filtered = sosfiltfilt(unipolar_filter, egm_unipolar, axis=0)

            # smooth the electrogram
            for channel_idx in range(egm_unipolar_filtered.shape[1]):
                electrogram = egm_unipolar_filtered[:, channel_idx].astype(float)

                smooth_window = 10
                smooth_kernel = np.ones(smooth_window) / smooth_window
                electrogram = np.convolve(electrogram, smooth_kernel, mode='same')

                egm_unipolar_filtered[:, channel_idx] = electrogram

            # unipolar electrograms before and after filtering
            debug_plot = 0
            if debug_plot: 
                result_folder = directory['result']
                mapping_name_unipolar = catheter['mapping_name_unipolar']
                utility.debug_plots.unipolar_electrogram_before_and_after_filtering(n, result_folder, egm_unipolar, egm_unipolar_filtered, mapping_name_unipolar)

            # create QRS morphology template for each of the unipolar electrograms
            # ------------------------------
            # copy the global QRS peak indices for each unipolar channel
            qrs_peak_indices_per_channel = [np.copy(qrs_peak_indices) for _ in range(egm_unipolar_filtered.shape[1])]

            # adaptive qrs template length based on the far-field morphology of the unipolar electrograms
            signal = egm_unipolar_filtered
            peak_indices = qrs_peak_indices_per_channel
            qrs_half_window = utility.signal_processing.estimate_far_field_half_window(signal,peak_indices)
            template_len = 2 * qrs_half_window + 1

            # create a median template only when aligned beats have consistent morphology
            half_window = qrs_half_window
            qrs_template_unipolar = utility.signal_processing.create_consistent_template(signal,peak_indices,half_window)

            # subtract the QRS template from each unipolar electrogram to remove the QRS component
            # ------------------------------
            qrs_subtracted = egm_unipolar_filtered.copy()
            qrs_taper_size = 10
            qrs_subtraction_window = np.ones(template_len, dtype=float)
            qrs_taper = 0.5 * (1 - np.cos(np.pi * np.arange(qrs_taper_size) / qrs_taper_size))
            qrs_subtraction_window[:qrs_taper_size] = qrs_taper
            qrs_subtraction_window[-qrs_taper_size:] = qrs_taper[::-1]

            for channel_idx in range(egm_unipolar_filtered.shape[1]):
                for peak_idx in qrs_peak_indices:
                    start = peak_idx - qrs_half_window
                    end = peak_idx + qrs_half_window + 1
                    seg_start = max(0, start)
                    seg_end = min(egm_unipolar_filtered.shape[0], end)

                    signal_segment = egm_unipolar_filtered[seg_start:seg_end, channel_idx]

                    if peak_idx <= qrs_half_window:
                        missing_front = qrs_half_window - peak_idx
                        template_segment = qrs_template_unipolar[channel_idx, missing_front:missing_front + signal_segment.shape[0]]
                    elif peak_idx + qrs_half_window + 1 > egm_unipolar_filtered.shape[0]:
                        template_segment = qrs_template_unipolar[channel_idx, :signal_segment.shape[0]]
                    else:
                        template_segment = qrs_template_unipolar[channel_idx, :]

                    if peak_idx <= qrs_half_window:
                        subtraction_window = qrs_subtraction_window[missing_front:missing_front + signal_segment.shape[0]]
                    elif peak_idx + qrs_half_window + 1 > egm_unipolar_filtered.shape[0]:
                        subtraction_window = qrs_subtraction_window[:signal_segment.shape[0]]
                    else:
                        subtraction_window = qrs_subtraction_window

                    # only subtract when needed
                    template_magnitude = np.ptp(template_segment)
                    signal_magnitude = np.ptp(signal_segment)
                    if template_magnitude > 0.2 and signal_magnitude > 0.2:  # only subtract if both template and signal have significant magnitude
                        qrs_subtracted[seg_start:seg_end, channel_idx] -= template_segment * subtraction_window # the subtraction is tapered to avoid abrupt changes by multiplying with a tapering window (subtraction_window)

            mapping_electrogram_unipolar_qrs_subtracted[n] = qrs_subtracted

            # plot the original, QRS template, and QRS-subtracted electrograms
            debug_plot = 0
            if debug_plot: 
                result_folder = directory['result']
                utility.debug_plots.unipolar_qrs_template_qrs_subtracted(n, result_folder, egm_unipolar, egm_unipolar_filtered, qrs_half_window, qrs_peak_indices, qrs_template_unipolar, qrs_subtracted)

            # activation time detection on the QRS-subtracted unipolar electrograms
            # ------------------------------
            # for each qrs subtracted unipolar electrogram, do auto correlation to find out its cycle length
            n_samples, n_channels = qrs_subtracted.shape
            cycle_length_unipolar = np.full(n_channels, np.nan, dtype=float)
            autocorrelations = [np.zeros(n_samples, dtype=float) for _ in range(n_channels)]
            for channel_idx in range(n_channels):
                electrogram = qrs_subtracted[:, channel_idx]

                if np.ptp(electrogram) >= 0.5: # mV
                    electrogram = electrogram - np.mean(electrogram)

                    # find peaks on the electrogram
                    peak_indices, peak_properties = find_peaks(electrogram, distance=200, height=0.08*np.ptp(electrogram))

                    # check if the electrogram has good signal quality by comparing the peak heights: good signal means the peak heights are relatively similar
                    peak_values = peak_properties['peak_heights']
                    good_signal_flag = int(
                        peak_values.size > 0
                        and np.min(peak_values) >= 0.5 * np.max(peak_values)
                    )

                    if good_signal_flag == 1:
                        # # clip the electrogram to within +/- mean(peak_values)
                        # mean_peak_value = np.mean(peak_values)
                        # electrogram = np.clip(electrogram, -mean_peak_value, mean_peak_value)

                        # find out the cycle length via autocorrelation
                        autocorrelation = np.correlate(electrogram, electrogram, mode='full')[n_samples - 1:]
                        autocorrelation /= autocorrelation[0]
                        autocorrelations[channel_idx] = autocorrelation

                        # blank out the first portion of the autocorrelation to avoid detecting the zero-lag peak and nearby peaks
                        autocorrelation[:150] = 0
                        candidate_lags, properties = find_peaks(autocorrelation,distance=150)

                        if len(candidate_lags) > 0:
                            candidate_lags = candidate_lags[candidate_lags != 1]
                            if len(candidate_lags) > 0:
                                # idx = np.argmax(autocorrelation[candidate_lags]) # choose the peak with the highest autocorrelation value as the cycle length
                                idx = 0 # choose the first peak as the cycle length
                                cycle_length_unipolar[channel_idx] = candidate_lags[idx]

            # the cycle length of this recording segment is the median of the cycle lengths of all unipolar electrograms
            cycle_lengths_per_segment[n] = cycle_length_unipolar
            cl = cycle_length_unipolar[np.isfinite(cycle_length_unipolar)] # remove NaN values
            cycle_length_this_segment = np.median(cl) if len(cl) > 0 else np.nan

            # plot the QRS-subtracted unipolar electrograms and their autocorrelations
            debug_plot = 0
            if debug_plot: 
                result_folder = directory['result']
                utility.debug_plots.qrs_subtracted_autocorrelation(n, result_folder, qrs_subtracted, autocorrelations, cycle_length_unipolar, mapping_name_unipolar)

            # for each qrs subtracted unipolar electrogram, find out the activation time by finding the peak of the maximum negative slope (dv/dt)
            if not np.isnan(cycle_length_this_segment):
                minimum_activation_distance = int(np.floor(0.9 * cycle_length_this_segment))
            elif np.isnan(cycle_length_this_segment):
                minimum_activation_distance = 150

            activation_times_unipolar = [[] for _ in range(n_channels)]
            for channel_idx in range(n_channels):
                electrogram = qrs_subtracted[:, channel_idx]
                negative_dvdt = -np.diff(electrogram, prepend=electrogram[0])

                if np.ptp(electrogram) >= 0.5: # mV
                    # find peaks on the electrogram
                    peak_indices, peak_properties = find_peaks(negative_dvdt, distance=minimum_activation_distance, height=0.08*np.ptp(negative_dvdt))

                    # check if the negative_dvdt has good signal quality by comparing the peak heights: good signal means the peak heights are relatively similar
                    peak_values = peak_properties['peak_heights']
                    good_signal_flag = int(
                        peak_values.size > 0
                        and np.min(peak_values) >= 0.2 * np.max(peak_values)
                    )

                    if good_signal_flag == 1:
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
                            activation_peaks = []

                        # remove activation peaks that are at the very beginning or very end of the signal, as they may be artifacts
                        activation_peaks = activation_peaks[(activation_peaks != 0) & (activation_peaks != qrs_subtracted.shape[0])]

                        activation_times_unipolar[channel_idx] = activation_peaks

            # plot the before and after QRS-subtracted unipolar electrograms and the detected activation times
            debug_plot = 0
            if debug_plot: 
                result_folder = directory['result']
                utility.debug_plots.activation_initial(n, result_folder, qrs_subtracted, egm_unipolar, activation_times_unipolar, surface_signal_smooth)

            # refine activation time detections
            # ------------------------------
            no_activation_electrode = 0
            for i in range(0,len(activation_times_unipolar)):
                if len(activation_times_unipolar[i]) == 0:
                    no_activation_electrode = no_activation_electrode + 1

            if no_activation_electrode != len(activation_times_unipolar):
                # adaptive activation template length based on the morphology of the unipolar electrograms
                signal = egm_unipolar_filtered # here use the egm with QRS
                peak_indices = activation_times_unipolar
                half_window = utility.signal_processing.estimate_far_field_half_window(signal,peak_indices)
                template_len = 2 * half_window + 1

                # for each unipolar electrogram, create morphology template from the detected activation times
                signal = qrs_subtracted # here use the egm with removed QRS
                activation_template_unipolar = utility.signal_processing.create_consistent_template(signal,peak_indices,half_window,consistency_threshold=0.4)

                # for each unipolar electrogram channel, crosscorrelate the activation template with the QRS-subtracted unipolar electrogram to refine the activation time detections
                activation_times_unipolar_refined = [[] for _ in range(n_channels)]
                for channel_idx in range(n_channels):
                    electrogram = qrs_subtracted[:, channel_idx]

                    template = activation_template_unipolar[channel_idx, :]

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
            elif no_activation_electrode == len(activation_times_unipolar):
                activation_times_unipolar_refined = activation_times_unipolar

            mapping_electrogram_unipolar_activation[n] = activation_times_unipolar_refined

            # plot the original electrograms, activation template, and refined activation times
            debug_plot = 0
            if debug_plot: 
                result_folder = directory['result']
                utility.debug_plots.activation_template_correlation(n, result_folder, egm_unipolar, qrs_subtracted, activation_times_unipolar_refined, half_window, activation_template_unipolar, surface_signal_smooth, surface_signal_sum)

        # find out the cycle length of all the unipolar electrograms across all recording segments
        # ------------------------------------------------------------
        all_cycle_lengths = np.concatenate(cycle_lengths_per_segment) # stack all the cycle lengths from each segment into a single array
        all_cycle_lengths = all_cycle_lengths[np.isfinite(all_cycle_lengths)] # remove NaN values
        median_cycle_length = np.median(all_cycle_lengths)

        catheter_processed = {}
        catheter_processed['surface_ecg_sum'] = surface_ecg_sum
        catheter_processed['mapping_electrogram_unipolar_qrs_subtracted'] = mapping_electrogram_unipolar_qrs_subtracted
        catheter_processed['mapping_electrogram_unipolar_activation'] = mapping_electrogram_unipolar_activation
        catheter_processed['cycle_lengths_per_segment'] = cycle_lengths_per_segment
        catheter_processed['cycle_length'] = median_cycle_length

        print(f'    median cycle length across all segments: {median_cycle_length} ms')

        # find out the best window of interest
        # ------------------------------------------------------------
        half_window_size_of_woi = median_cycle_length*0.8 // 2

        electrode_positions_all = np.concatenate([
            np.asarray(segment, dtype=float).reshape(-1, 3)
            for segment in catheter['mapping_position_unipolar']
        ])

        # grab activations within window of interest
        t_range = range(2000-200, 2000+100)
        outlier_ratio = np.zeros((len(t_range),))
        for i in range(len(t_range)):
            t = t_range[i]
            t_start = t - half_window_size_of_woi # window of interest start time index
            t_end = t + half_window_size_of_woi # window of interest end time index
            catheter_processed = utility.ui_functions.grab_activations_within_window_of_interest(catheter_processed,t_start,t_end)

            # Find activation times that disagree with nearby electrodes.
            activation_segments = catheter_processed['mapping_electrogram_unipolar_activation_within_woi']
            activation_times_all = np.concatenate([
                np.asarray(segment, dtype=float).reshape(-1) for segment in activation_segments
            ])

            if len(activation_times_all) != len(electrode_positions_all):
                raise ValueError('Activation times and electrode positions must have the same length')

            valid = (
                np.isfinite(activation_times_all) & (activation_times_all > 0)
                & np.all(np.isfinite(electrode_positions_all), axis=1)
            )
            valid_indices = np.flatnonzero(valid)
            valid_times = activation_times_all[valid_indices]
            valid_positions = electrode_positions_all[valid_indices]
            outlier_mask = np.zeros(len(activation_times_all), dtype=bool)
            if len(valid_indices) >= 3:
                nearby_electrodes = cKDTree(valid_positions)
                for local_idx, position in enumerate(valid_positions):
                    neighbor_ids = [
                        idx for idx in nearby_electrodes.query_ball_point(position, r=10.0)
                        if idx != local_idx
                    ]
                    if len(neighbor_ids) >= 2:
                        neighbor_times = valid_times[neighbor_ids]
                        local_median = np.median(neighbor_times)
                        if (
                            abs(valid_times[local_idx] - local_median) > 10
                            and np.count_nonzero(abs(neighbor_times - local_median) <= 10) >= 2
                        ):
                            outlier_mask[valid_indices[local_idx]] = True

            outliers_id = np.flatnonzero(outlier_mask)

            outlier_ratio[i] = len(outliers_id) / len(valid_times)

        min_outlier_ratio_idx = np.argmin(outlier_ratio)

        t_start = t_range[min_outlier_ratio_idx] - half_window_size_of_woi # window of interest start time index
        t_end = t_range[min_outlier_ratio_idx] + half_window_size_of_woi # window of interest end time index
        catheter_processed = utility.ui_functions.grab_activations_within_window_of_interest(catheter_processed,t_start,t_end)

        # # keep each segment's channel activations together even when channel counts differ.
        # activation_array = np.empty(n_segment, dtype=object) # to make npsavez 
        # for segment_id, segment_activations in enumerate(catheter_processed['mapping_electrogram_unipolar_activation']):
        #     activation_array[segment_id] = segment_activations
        # catheter_processed['mapping_electrogram_unipolar_activation'] = activation_array

        # project electrode positions onto the mesh surface
        # ------------------------------------------------------------
        mesh = catheter['mesh'].item()

        original_vertex = mesh['geometry_original_vertex']
        original_face = mesh['geometry_original_face']
        refined_vertex = mesh['vertex']
        refined_face = mesh['face']

        electrode_positions = []
        for n in range(len(catheter['mapping_position_unipolar'])):
            positions = catheter['mapping_position_unipolar'][n]
            electrode_positions.append(positions)

        print('    project electrodes to original and refined meshes')
        electrode_positions_all = np.concatenate(
            [np.asarray(positions, dtype=float).reshape(-1, 3) for positions in electrode_positions],
            axis=0,
        )
        projection_normal_vicinity = 10.0 # mesh coordinate units; electrodes farther from every nearby vertex-normal line are rejected
        projection_max_distance = 20.0 # mesh coordinate units along the bidirectional vertex-normal line
        electrode_positions_on_original_mesh_all, _ = utility.ui_functions.project_electrodes_to_mesh(
            original_vertex,
            original_face,
            electrode_positions_all,
            normal_vicinity=projection_normal_vicinity,
            projection_max_distance=projection_max_distance,
        )
        electrode_positions_on_refined_mesh_all, _ = utility.ui_functions.project_electrodes_to_mesh(
            refined_vertex,
            refined_face,
            electrode_positions_all,
            normal_vicinity=projection_normal_vicinity,
            projection_max_distance=projection_max_distance,
        )

        original_rejected_count = np.count_nonzero(
            ~np.all(np.isfinite(electrode_positions_on_original_mesh_all), axis=1)
        )
        refined_rejected_count = np.count_nonzero(
            ~np.all(np.isfinite(electrode_positions_on_refined_mesh_all), axis=1)
        )

        catheter_processed['geometry_original_vertex'] = original_vertex
        catheter_processed['geometry_original_face'] = original_face
        catheter_processed['vertex'] = refined_vertex
        catheter_processed['face'] = refined_face
        catheter_processed['electrode_positions'] = electrode_positions
        catheter_processed['electrode_positions_on_original_mesh_all'] = electrode_positions_on_original_mesh_all
        catheter_processed['electrode_positions_on_refined_mesh_all'] = electrode_positions_on_refined_mesh_all

        # savez will have error if channel counts differ between segments, need to convert to object type
        for field_name, field_values in catheter_processed.items():
            if not isinstance(field_values, (list, tuple)):
                continue
            try:
                np.asarray(field_values)
            except ValueError:
                object_array = np.empty(len(field_values), dtype=object)
                object_array[:] = field_values
                catheter_processed[field_name] = object_array

        file_path = directory['data'] / f'{name_prefix}_carto_catheter_processed.npz'
        np.savez(file_path, **catheter_processed)

        # plot the electrode positions and the projections on the mesh surface
        debug_plot = 0
        if debug_plot == 1: 
            utility.debug_plots.electrode_position_and_projection(n, original_vertex, original_face, electrode_positions_all, electrode_positions_on_original_mesh_all, refined_vertex, refined_face, electrode_positions_on_refined_mesh_all)

#%%
print('done')
