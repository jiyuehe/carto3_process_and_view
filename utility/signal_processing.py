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
import matplotlib.pyplot as plt

# estimate a common template half-width from the far-field envelope
def estimate_far_field_half_window(signal,peak_indices):
    minimum_half_window = 50
    maximum_half_window = 120

    if peak_indices is None:
        half_window = minimum_half_window

    # compute per-channel median envelopes from their peaks
    n_samples, n_channels = signal.shape
    initial_half_window = maximum_half_window
    envelopes_per_channel = []
    for ch in range(n_channels):
        ch_peaks = peak_indices[ch]
        ch_envelopes = []
        for peak_idx in ch_peaks:
            peak_idx = int(peak_idx)
            start = max(0, peak_idx - initial_half_window)
            end = min(n_samples, peak_idx + initial_half_window + 1)
            beat = signal[start:end, ch]

            if beat.shape[0] < 2 * initial_half_window + 1: # the beat is too close to the edge of the signal, pad with zeros
                padded = np.zeros((2 * initial_half_window + 1,), dtype=float)
                if peak_idx <= initial_half_window:
                    missing_front = initial_half_window - peak_idx
                    padded[missing_front:missing_front + beat.shape[0]] = beat
                else:
                    padded[:beat.shape[0]] = beat
                beat = padded

            beat = beat - np.median(beat)

            # per-peak envelope for this channel is absolute amplitude over time
            ch_envelopes.append(np.abs(beat))

        if len(ch_envelopes) > 0:
            # median of the ch_envelopes
            envelopes_per_channel.append(np.median(np.stack(ch_envelopes, axis=0), axis=0))

    if len(envelopes_per_channel) == 0:
        half_window = minimum_half_window

    debug_plot = 0
    if debug_plot:
        signal_range = np.nanmax(np.ptp(signal[:, :n_channels], axis=0)) 
        trace_spacing = max(1.0, float(signal_range) * 1.5)
        channel_offsets = np.arange(n_channels) * trace_spacing

        fig, (signal_ax, envelope_ax) = plt.subplots(
            1,
            2,
            figsize=(14, max(5, 0.5 * n_channels)),
            sharey=True,
            gridspec_kw={'width_ratios': [10, 1]},
        )

        for ch in range(n_channels):
            offset = channel_offsets[ch]
            signal_ax.plot(signal[:, ch] + offset, color='blue', linewidth=0.8)
            envelope_ax.plot(envelopes_per_channel[ch] + offset, color='blue', linewidth=0.8)

        signal_ax.set_title('Signals')
        envelope_ax.set_title('Median envelopes')
        signal_ax.set_xlim(0, signal.shape[0] - 1)
        envelope_ax.set_xlim(0, envelopes_per_channel[0].shape[0] - 1)
        y_min = channel_offsets[0] - trace_spacing * 0.5
        y_max = channel_offsets[-1] + trace_spacing * 0.5
        signal_ax.set_ylim(y_min, y_max)
        envelope_ax.set_ylim(y_min, y_max)
        fig.tight_layout()
        plt.show()

    envelope = np.median(np.stack(envelopes_per_channel, axis=0), axis=0) # the median envelope of all channels

    # estimate background level from the edges
    # a real QRS/far-field peak is near the center
    # the edges should be mostly quiet
    # the median and MAD of those edge samples estimate the noise floor
    # this is a standard robust baseline estimate.
    edge_samples = max(5, initial_half_window // 5)
    edge_values = np.concatenate((envelope[:edge_samples], envelope[-edge_samples:]))
    baseline = np.median(edge_values)
    noise = 1.4826 * np.median(np.abs(edge_values - baseline))

    # create a threshold that is above the baseline and noise, but below the peak
    # anything above this threshold is considered “signal-like”
    peak_excess = max(0.0, np.max(envelope) - baseline)
    threshold = max(baseline + 2 * noise, baseline + 0.1 * peak_excess)

    def find_boundary(start, step): 
        # walks left/right until it sees a run of samples below threshold
        # Once enough consecutive samples are below threshold, it treats that as the edge of the main peak
        index = start
        below_threshold_count = 0
        required_below_threshold = 5

        while 0 < index < envelope.shape[0] - 1:
            index += step
            if envelope[index] < threshold:
                below_threshold_count += 1
                if below_threshold_count >= required_below_threshold:
                    return index - step * (required_below_threshold - 1)
            else:
                below_threshold_count = 0

        return index

    # find where the envelope falls back below threshold
    center = initial_half_window
    if envelope[center] < threshold:
        half_window = minimum_half_window

    left = find_boundary(center, -1)
    right = find_boundary(center, 1)
    estimated_half_window = max(center - left, right - center) # find out how wide is the main signal bump around the peak, measured in samples

    # clamp to allowed range
    half_window = int(np.clip(estimated_half_window, minimum_half_window, maximum_half_window))

    return half_window

# create a median template only when aligned beats have consistent morphology
def create_consistent_template(signal, peak_indices, half_window, consistency_threshold = 0.6):
    n_samples, n_channels = signal.shape
    template_len = 2 * half_window + 1
    template = np.zeros((n_channels, template_len), dtype=float)

    for ch in range(n_channels):
        ch_peak_indices = np.asarray(peak_indices[ch], dtype=int)
        raw_beats = []
        for peak_idx in ch_peak_indices:
            start = max(0, peak_idx - half_window)
            end = min(n_samples, peak_idx + half_window + 1)
            beat = signal[start:end, ch]

            if beat.shape[0] < template_len:
                padded = np.zeros((template_len,), dtype=float)
                if peak_idx <= half_window:
                    missing_front = half_window - peak_idx
                    padded[missing_front:missing_front + beat.shape[0]] = beat
                else:
                    padded[:beat.shape[0]] = beat
                beat = padded

            beat = beat - np.median(beat)

            raw_beats.append(beat)

        # if no beats were collected for this channel, leave the template as zeros
        if len(raw_beats) == 0:
            template[ch, :] = 0.0
            continue

        beat_stack = np.stack(raw_beats, axis=0)
        morphology_median = np.median(beat_stack, axis=0)

        # compute the correlation of each beat with the median morphology
        beat_correlations = []
        for beat in beat_stack:
            beat_centered = beat - np.median(beat)
            median_centered = morphology_median - np.median(morphology_median)
            denom = np.linalg.norm(beat_centered) * np.linalg.norm(median_centered)
            if denom > 0:
                corr = float(np.dot(beat_centered, median_centered) / denom)
            else:
                corr = 0.0
            beat_correlations.append(corr)

        # check which beats are consistent with the median morphology
        beat_correlations = np.asarray(beat_correlations, dtype=float)
        beat_correlations = np.nan_to_num(beat_correlations, nan=0.0)
        beat_correlations_mean = np.mean(beat_correlations)

        # keep a template for each channel instead of overwriting a single shared variable
        if beat_correlations_mean >= consistency_threshold:
            template[ch, :] = morphology_median
        else:
            template[ch, :] = 0.0

    return template
