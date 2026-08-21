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

def create_consistent_template(signal, peak_indices, half_window, consistency_threshold=0.7):
    # Create a median template only when aligned beats have consistent morphology
    if signal.ndim == 1:
        signal = signal[:, np.newaxis] # it adds a new axis to a 1D array so it becomes a column vector, so that "n_samples, n_channels = signal.shape" will not have an error

    n_samples, n_channels = signal.shape
    template_len = 2 * half_window + 1
    zero_template = np.zeros((n_channels, template_len), dtype=float)

    if len(peak_indices) == 0:
        return np.squeeze(zero_template), False

    raw_beats = []
    normalized_beats = []
    for peak_idx in peak_indices:
        start = max(0, peak_idx - half_window)
        end = min(n_samples, peak_idx + half_window + 1)
        beat = signal[start:end, :]

        if beat.shape[0] < template_len:
            padded = np.zeros((template_len, n_channels), dtype=float)
            if peak_idx <= half_window:
                missing_front = half_window - peak_idx
                padded[missing_front:missing_front + beat.shape[0], :] = beat
            else:
                padded[:beat.shape[0], :] = beat
            beat = padded

        beat = beat - np.median(beat, axis=0, keepdims=True)
        beat_norm = np.linalg.norm(beat)
        if beat_norm == 0:
            return np.squeeze(zero_template), False
        raw_beats.append(beat)
        normalized_beats.append(beat / beat_norm)

    if len(raw_beats) == 1:
        return np.squeeze(raw_beats[0].T), True

    beat_stack = np.stack(normalized_beats, axis=0)
    morphology = np.median(beat_stack, axis=0)
    morphology_norm = np.linalg.norm(morphology)
    if morphology_norm == 0:
        return np.squeeze(zero_template), False

    morphology_reference = morphology / morphology_norm
    morphology_correlations = np.sum(
        beat_stack * morphology_reference,
        axis=(1, 2),
    )
    if np.any(morphology_correlations < consistency_threshold):
        return np.squeeze(zero_template), False

    return np.squeeze(np.median(np.stack(raw_beats, axis=0), axis=0).T), True

def estimate_far_field_half_window(signal,peak_indices,minimum_half_window=40,maximum_half_window=120,edge_margin=10):
    # estimate a common template half-width from the unipolar far-field envelope
    if len(peak_indices) == 0:
        return minimum_half_window

    n_samples, n_channels = signal.shape
    envelopes = []
    initial_half_window = maximum_half_window

    for peak_idx in peak_indices:
        start = max(0, peak_idx - initial_half_window)
        end = min(n_samples, peak_idx + initial_half_window + 1)
        beat = signal[start:end, :]

        if beat.shape[0] < 2 * initial_half_window + 1:
            padded = np.zeros((2 * initial_half_window + 1, n_channels), dtype=float)
            if peak_idx <= initial_half_window:
                missing_front = initial_half_window - peak_idx
                padded[missing_front:missing_front + beat.shape[0], :] = beat
            else:
                padded[:beat.shape[0], :] = beat
            beat = padded

        beat = beat - np.median(beat, axis=0, keepdims=True)
        envelopes.append(np.percentile(np.abs(beat), 75, axis=1))

    envelope = np.median(np.stack(envelopes, axis=0), axis=0)
    edge_samples = max(5, initial_half_window // 5)
    edge_values = np.concatenate((envelope[:edge_samples], envelope[-edge_samples:]))
    baseline = np.median(edge_values)
    noise = 1.4826 * np.median(np.abs(edge_values - baseline))
    peak_excess = max(0.0, np.max(envelope) - baseline)
    # Use a low tail threshold so lower-amplitude far-field activity is retained.
    threshold = max(baseline + 2 * noise, baseline + 0.1 * peak_excess)

    center = initial_half_window
    if envelope[center] < threshold:
        return minimum_half_window

    def find_boundary(start, step):
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

    left = find_boundary(center, -1)
    right = find_boundary(center, 1)
    estimated_half_window = max(center - left, right - center) + edge_margin
    return int(np.clip(estimated_half_window, minimum_half_window, maximum_half_window))
