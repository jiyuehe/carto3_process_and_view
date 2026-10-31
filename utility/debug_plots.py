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

import numpy as np
import matplotlib.pyplot as plt

def mapping_and_surface_electrogram(segment_id, result_folder, mapping_name_unipolar, mapping_electrogram_unipolar, surface_name, surface_electrogram):
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharex=True)

    title = 'Mapping unipolar electrograms'
    recordings = mapping_electrogram_unipolar[segment_id]
    names = mapping_name_unipolar[segment_id]
    axis = axes[0]

    channel_magnitude = np.ptp(recordings, axis=0)
    trace_spacing = max(1.0, np.nanmean(channel_magnitude) * 1.05)
    trace_offsets = np.arange(recordings.shape[1]) * trace_spacing

    axis.plot(
        np.arange(recordings.shape[0]),
        recordings + trace_offsets,
        color='blue',
        linewidth=1,
    )
    axis.set_title(title)
    axis.set_yticks(trace_offsets)
    axis.set_yticklabels(names)
    axis.grid(False)
    axis.set_xlabel('Time (ms)')

    title = 'Surface electrograms'
    recordings = surface_electrogram[segment_id]
    names = surface_name
    axis = axes[1]

    channel_magnitude = np.ptp(recordings, axis=0)
    trace_spacing = max(1.0, np.nanmean(channel_magnitude) * 1.05)
    trace_offsets = np.arange(recordings.shape[1]) * trace_spacing

    axis.plot(
        np.arange(recordings.shape[0]),
        recordings + trace_offsets,
        color='blue',
        linewidth=1,
    )
    axis.set_title(title)
    axis.set_yticks(trace_offsets)
    axis.set_yticklabels(names)
    axis.grid(False)
    axis.set_xlabel('Time (ms)')

    fig.suptitle(
        f'Recording segment {segment_id} of '
        f'[0, {len(mapping_electrogram_unipolar) - 1}]'
    )
    fig.tight_layout()

    # save the figure
    fig_path = result_folder / f'1_recording_segment_{segment_id}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def QRS_timings_on_surface_electrograms(n, result_folder, surface_name, egm_surface, egm_surface_filtered, surface_signal_sum, surface_signal_smooth, qrs_peak_indices):
    fig, ax = plt.subplots(figsize=(8, 8))
    n_channel = egm_surface.shape[1]

    # determine spacing based on signal magnitudes to separate traces visually
    per_channel_magnitude = np.ptp(egm_surface, axis=0) # peak-to-peak voltage
    max_magnitude = np.nanmean(per_channel_magnitude) 
    spacing = max_magnitude * 1.5
    offsets = np.arange(n_channel) * spacing

    # plot each channel with an offset
    for channel_idx in range(n_channel):
        ax.plot(egm_surface[:, channel_idx] + offsets[channel_idx], color='magenta', linewidth=1)
        ax.plot(egm_surface_filtered[:, channel_idx] + offsets[channel_idx], color='blue', linewidth=1)

    # plot the summed signal scaled and placed at bottom
    surface_signal_sum_scaled = surface_signal_sum * (max_magnitude / np.ptp(surface_signal_sum)) - spacing
    surface_signal_smooth_scaled = surface_signal_smooth * (max_magnitude / np.ptp(surface_signal_smooth)) - spacing

    ax.plot(surface_signal_sum_scaled, color='magenta', linewidth=1)
    ax.plot(surface_signal_smooth_scaled, color='blue', linewidth=1)
    ax.scatter(qrs_peak_indices, surface_signal_smooth_scaled[qrs_peak_indices], color='red', s=20, zorder=5)
    
    ax.set_xlabel('Time (ms)')
    yticks = np.concatenate(([-spacing], offsets))
    ylabels = ['sum'] + list(surface_name)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title(f'Surface ECGs and QRS detection (segment id {n})')
    plt.tight_layout()

    # save the figure
    fig_path = result_folder / f'2_qrs_timing_{n}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

