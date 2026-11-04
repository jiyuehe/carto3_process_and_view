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

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

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

def qrs_timings_on_surface_electrograms(n, result_folder, surface_name, egm_surface, egm_surface_filtered, surface_signal_sum, surface_signal_smooth, qrs_peak_indices):
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
    ax.scatter(
        qrs_peak_indices,
        surface_signal_smooth_scaled[qrs_peak_indices],
        facecolors=(1, 0, 0, 0.6),
        edgecolors='none',
        s=20,
        zorder=5,
    )
    
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

def unipolar_electrogram_before_and_after_filtering(n, result_folder, egm_unipolar, egm_unipolar_filtered, mapping_name_unipolar):
    n_channels = egm_unipolar.shape[1]
    time_ms = np.arange(egm_unipolar.shape[0])
    channel_magnitude = np.maximum(
        np.ptp(egm_unipolar, axis=0),
        np.ptp(egm_unipolar_filtered, axis=0),
    )
    trace_spacing = max(1.0, np.nanmean(channel_magnitude) * 1.05)
    trace_offsets = np.arange(n_channels) * trace_spacing

    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)
    for axis, electrograms, title in zip(
        axes,
        (egm_unipolar, egm_unipolar_filtered),
        ('Before filtering', 'After filtering'),
    ):
        axis.plot(time_ms, electrograms + trace_offsets, linewidth=1,color='blue')
        axis.set_title(title)
        axis.set_yticks(trace_offsets)
        axis.set_yticklabels(mapping_name_unipolar[n], fontsize=8)
        axis.grid(False)

    axes[0].set_xlabel('Time (ms)')
    axes[1].set_xlabel('Time (ms)')
    fig.suptitle(f'Unipolar electrogram (segment id {n})')
    fig.tight_layout()

    # save the figure
    fig_path = result_folder / f'3_unipolar_bandpass_filter_{n}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def unipolar_qrs_template_qrs_subtracted(n, result_folder, egm_unipolar, egm_unipolar_filtered, qrs_half_window, qrs_peak_indices, qrs_template_unipolar, qrs_subtracted):
    n_channels = egm_unipolar_filtered.shape[1]
    template_x = np.arange(-qrs_half_window, qrs_half_window + 1) # x-axis for the QRS template

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(1, 3, width_ratios=[20, 1, 20], wspace=0.08)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    channel_magnitude = np.maximum(
        np.ptp(egm_unipolar, axis=0),
        np.ptp(egm_unipolar_filtered, axis=0),
    )

    y_spacing = max(1.0, np.nanmean(channel_magnitude) * 1.05)
    y_offset = np.arange(n_channels) * y_spacing

    if len(qrs_peak_indices) > 0:
        for peak_idx in qrs_peak_indices:
            ax_left.axvline(peak_idx, color='red', linewidth=1.0, clip_on=True) # the vertical line indicating QRS timing

    for channel_idx in range(n_channels):
        signal_trace = egm_unipolar_filtered[:, channel_idx] + y_offset[channel_idx]
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
        stacked_trace_values.append(egm_unipolar_filtered[:, channel_idx] + y_offset[channel_idx])
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

    fig_path = result_folder / f'4_qrs_subtraction_{n}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def qrs_subtracted_autocorrelation(n, result_folder, qrs_subtracted, autocorrelations, cycle_length_unipolar, mapping_name_unipolar):
    n_samples, n_channels = qrs_subtracted.shape

    time_ms = np.arange(n_samples)
    signal_magnitude = np.ptp(qrs_subtracted, axis=0)
    trace_spacing = max(1.0, np.nanmean(signal_magnitude) * 1.05)
    trace_offsets = np.arange(n_channels) * trace_spacing
    autocorrelation_scale = trace_spacing

    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    axes[0].plot(
        time_ms,
        qrs_subtracted + trace_offsets,
        linewidth=1,
        color='blue',
    )
    axes[0].set_title('QRS-subtracted unipolar electrograms')

    autocorrelation_traces = np.column_stack(autocorrelations)
    axes[1].plot(
        time_ms,
        autocorrelation_traces * autocorrelation_scale + trace_offsets,
        linewidth=1,
        color='blue',
    )
    for channel_idx in range(n_channels):
        if np.isfinite(cycle_length_unipolar[channel_idx]):
            cl = int(cycle_length_unipolar[channel_idx])
            axes[1].scatter(
                time_ms[cl],
                autocorrelations[channel_idx][cl] * autocorrelation_scale
                + trace_offsets[channel_idx],
                facecolors=(1, 0, 0, 0.6),
                edgecolors='none',
                s=16,
                zorder=3,
            )
    axes[1].set_title('Autocorrelations')

    for axis in axes:
        axis.set_yticks(trace_offsets)
        axis.set_yticklabels(mapping_name_unipolar[n], fontsize=8)
        axis.grid(False)
        axis.autoscale(tight=True)

    axes[0].set_xlabel('Time (ms)')
    axes[1].set_xlabel('Lag (ms)')
    fig.suptitle(f'find out cycle length via autocorrelation (segment id {n})')
    fig.tight_layout()

    # save the figure
    fig_path = result_folder / f'5_autocorrelation_{n}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def activation_initial(n, result_folder, qrs_subtracted, negative_dvdt, egm_unipolar, activation_times_unipolar, surface_signal_smooth):
    n_samples, n_channels = qrs_subtracted.shape

    sample_axis = np.arange(n_samples)
    trace_range = max(
        np.nanmean(np.ptp(egm_unipolar, axis=0)),
        np.nanmean(np.ptp(qrs_subtracted, axis=0)),
    )
    trace_spacing = max(1.0, trace_range * 1.05)
    trace_offsets = np.arange(n_channels) * trace_spacing

    surface_signal_scaled = surface_signal_smooth * (trace_range / np.ptp(surface_signal_smooth)) - trace_spacing

    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharex=True)
    original_axis, subtracted_axis = axes

    for channel_idx in range(n_channels):
        offset = trace_offsets[channel_idx]
        original_axis.plot(
            sample_axis,
            egm_unipolar[:, channel_idx] + offset,
            color='blue',
            linewidth=0.7,
        )

        channel_negative_dvdt = negative_dvdt[:, channel_idx]
        dvdt_peak = np.max(np.abs(channel_negative_dvdt))
        scaled_negative_dvdt = channel_negative_dvdt * (
            0.3 * trace_spacing / max(dvdt_peak, np.finfo(float).eps)
        )
        subtracted_axis.plot(
            sample_axis,
            scaled_negative_dvdt + offset,
            color='magenta',
            linewidth=0.7,
            zorder=1,
        )
        subtracted_axis.plot(
            sample_axis,
            qrs_subtracted[:, channel_idx] + offset,
            color='blue',
            linewidth=0.7,
            zorder=2,
        )
        activation_peaks = activation_times_unipolar[channel_idx]
        subtracted_axis.scatter(
            activation_peaks,
            qrs_subtracted[activation_peaks, channel_idx] + offset,
            facecolors=(1, 0, 0, 0.6),
            edgecolors='none',
            s=20,
            zorder=3,
        )

    surface_axis = np.arange(n_samples)
    original_axis.plot(surface_axis, surface_signal_scaled, color='magenta', linewidth=1.0)
    subtracted_axis.plot(surface_axis, surface_signal_scaled, color='magenta', linewidth=1.0)

    original_axis.set_title('Original unipolar electrograms')
    subtracted_axis.set_title('QRS-subtracted electrograms, negative dV/dt, and activation times')
    original_axis.set_xlabel('time (ms)')
    subtracted_axis.set_xlabel('time (ms)')
    original_axis.set_yticks([])
    subtracted_axis.set_yticks([])
    original_axis.set_xlim(sample_axis[0], sample_axis[-1])

    # set the y-axis limits
    top_unipolar_egm = qrs_subtracted[:, -1] + trace_offsets[-1]
    y_min = np.nanmin(surface_signal_scaled)
    y_max = np.nanmax(top_unipolar_egm)
    original_axis.set_ylim(y_min, y_max)
    subtracted_axis.set_ylim(y_min, y_max)

    fig.tight_layout()

    # save the figure
    fig_path = result_folder / f'6_activation_initial_{n}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def activation_template_correlation(n, result_folder, egm_unipolar, qrs_subtracted, activation_times_unipolar_refined, half_window, activation_template_unipolar, cross_correlation, surface_signal_smooth, surface_signal_sum):
    n_samples, n_channels = qrs_subtracted.shape

    sample_axis = np.arange(n_samples)
    trace_range = max(
        np.nanmean(np.ptp(egm_unipolar, axis=0)),
        np.nanmean(np.ptp(qrs_subtracted, axis=0)),
    )
    trace_spacing = max(1.0, trace_range * 1.05)
    trace_offsets = np.arange(n_channels) * trace_spacing

    template_axis_offset = np.arange(n_channels) * trace_spacing
    template_x = np.arange(-half_window, half_window + 1)
    surface_signal_scaled = surface_signal_smooth * (trace_range / np.ptp(surface_signal_smooth)) - trace_spacing

    fig, axes = plt.subplots(1, 3, figsize=(12, 8), gridspec_kw={'width_ratios': [30, 1, 30]})
    original_axis, template_axis, subtracted_axis = axes

    for channel_idx in range(n_channels):
        offset = trace_offsets[channel_idx]
        original_axis.plot(
            sample_axis,
            egm_unipolar[:, channel_idx] + offset,
            color='blue',
            linewidth=0.7,
            zorder=1,
        )
        channel_correlation = cross_correlation[:, channel_idx]
        correlation_peak = np.max(np.abs(channel_correlation))
        scaled_correlation = channel_correlation * (
            0.3 * trace_spacing / max(correlation_peak, np.finfo(float).eps)
        )

        subtracted_axis.plot(
            sample_axis,
            scaled_correlation + offset,
            color='magenta',
            linewidth=0.7,
            zorder=1,
        )
        subtracted_axis.plot(
            sample_axis,
            qrs_subtracted[:, channel_idx] + offset,
            color='blue',
            linewidth=0.7,
            zorder=2,
        )
        activation_peaks = activation_times_unipolar_refined[channel_idx]
        if not np.array_equal(activation_peaks, [0]):
            subtracted_axis.scatter(
                activation_peaks,
                qrs_subtracted[activation_peaks, channel_idx] + offset,
                facecolors=(1, 0, 0, 0.6),
                edgecolors='none',
                s=20,
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
    subtracted_axis.set_title('QRS-subtracted electrograms, cross-correlation, and activation times')
    original_axis.set_xlabel('time (ms)')
    template_axis.set_xlabel('time (ms)')
    subtracted_axis.set_xlabel('time (ms)')
    original_axis.set_yticks([])
    template_axis.set_yticks([])
    subtracted_axis.set_yticks([])
    original_axis.set_xlim(sample_axis[0], sample_axis[-1])
    template_axis.set_xlim(template_x[0], template_x[-1])
    subtracted_axis.set_xlim(sample_axis[0], sample_axis[-1])

    # set the y-axis limits
    top_unipolar_egm = egm_unipolar[:, -1] + trace_offsets[-1]
    y_min = np.nanmin(surface_signal_scaled)
    y_max = np.nanmax(top_unipolar_egm)
    original_axis.set_ylim(y_min, y_max)
    subtracted_axis.set_ylim(y_min, y_max)
    template_axis.set_ylim(y_min, y_max)

    fig.tight_layout()

    fig_path = result_folder / f'7_activation_refined_{n}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def electrode_position_and_projection(n, original_vertex, original_face, electrode_positions_all, electrode_positions_on_original_mesh_all, refined_vertex, refined_face, electrode_positions_on_refined_mesh_all):
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=original_vertex[:, 0], y=original_vertex[:, 1], z=original_vertex[:, 2],
        i=original_face[:, 0], j=original_face[:, 1], k=original_face[:, 2],
        color='lightgray', opacity=0.5, name='Mesh', hoverinfo='skip'
    ))
    p = electrode_positions_all
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
        marker=dict(
            size=4,
            color='red',
            opacity=0.6,
            line=dict(width=0),
        ), name='Original electrodes',
        legendgroup='original', showlegend=(n == 0)
    ))
    p = electrode_positions_on_original_mesh_all
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
        marker=dict(size=4, color='blue'), name='Projected electrodes',
        legendgroup='projected', showlegend=(n == 0)
    ))
    fig.update_layout(
        title='Electrode positions projected onto original mesh surface',
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', dragmode='orbit'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=refined_vertex[:, 0], y=refined_vertex[:, 1], z=refined_vertex[:, 2],
        i=refined_face[:, 0], j=refined_face[:, 1], k=refined_face[:, 2],
        color='lightgray', opacity=0.5, name='Mesh', hoverinfo='skip'
    ))
    p = electrode_positions_all
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
        marker=dict(
            size=4,
            color='red',
            opacity=0.6,
            line=dict(width=0),
        ), name='Original electrodes',
        legendgroup='original', showlegend=(n == 0)
    ))
    p = electrode_positions_on_refined_mesh_all
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
        marker=dict(size=4, color='blue'), name='Projected electrodes',
        legendgroup='projected', showlegend=(n == 0)
    ))
    fig.update_layout(
        title='Electrode positions projected onto refined mesh surface',
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', dragmode='orbit'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()
