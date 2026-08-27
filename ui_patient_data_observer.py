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
from scipy.interpolate import griddata
from scipy.signal import find_peaks
from flask import Flask, render_template, jsonify, request
import os
import tempfile
import webbrowser
import threading
import subprocess
import time
import configuration
import utility


def estimate_electrogram_cycle_lengths(electrogram_segments):
    """Estimate one cycle length per electrogram from normalized autocorrelation."""
    cycle_lengths = []

    for segment in electrogram_segments:
        electrograms = np.asarray(segment, dtype=np.float64)
        if electrograms.ndim != 2 or electrograms.shape[0] < 3:
            continue

        n_samples, _ = electrograms.shape
        finite_channels = np.all(np.isfinite(electrograms), axis=0)
        active_channels = finite_channels & (np.ptp(electrograms, axis=0) >= 0.3)
        if not np.any(active_channels):
            continue

        signals = electrograms[:, active_channels]
        signals = signals - np.mean(signals, axis=0, keepdims=True)

        # FFT equivalent of the one-sided np.correlate calculation already
        # used in process_electrogram.py, evaluated for all channels together.
        fft_length = 1 << (2 * n_samples - 2).bit_length()
        spectra = np.fft.rfft(signals, n=fft_length, axis=0)
        autocorrelations = np.fft.irfft(
            spectra * np.conjugate(spectra), n=fft_length, axis=0
        )[:n_samples]

        zero_lag = autocorrelations[0]
        usable = np.isfinite(zero_lag) & (zero_lag > 0)
        autocorrelations[:, usable] /= zero_lag[usable]

        for channel_index in np.flatnonzero(usable):
            autocorrelation = autocorrelations[:, channel_index]
            candidate_lags, _ = find_peaks(autocorrelation, prominence=0.05)
            candidate_lags = candidate_lags[candidate_lags != 1]
            if candidate_lags.size:
                strongest = candidate_lags[np.argmax(autocorrelation[candidate_lags])]
                cycle_lengths.append(float(strongest))

    return np.asarray(cycle_lengths, dtype=np.float64)

#%%
# setting
directory = configuration.directory_setup()
name_prefix = configuration.map_name()

# load map data
data = np.load(directory['data'] / f'{name_prefix}_mesh.npz', allow_pickle=True)
mesh = {
    k: (data[k].item() if isinstance(data[k], np.ndarray) and data[k].ndim == 0 else data[k])
    for k in data.files
}

data = np.load(directory['data'] / f'{name_prefix}_catheter.npz', allow_pickle=True)
catheter = {
    k: (data[k].item() if isinstance(data[k], np.ndarray) and data[k].ndim == 0 else data[k])
    for k in data.files
}

# variable to store data
electrode_positions = np.asarray(mesh['electrode_positions'], dtype=object) # shape is (n_segments,)
segment_count = len(electrode_positions)
segment_electrode_count = int(electrode_positions[0].shape[0])
egm_uni_original = np.asarray(catheter.get('mapping_electrogram_unipolar', []), dtype=object) # shape is (n_segments, n_samples, n_electrodes)
egm_uni_qrs_subtracted = np.asarray(catheter.get('mapping_electrogram_unipolar_qrs_subtracted', []), dtype=object) # shape is (n_segments, n_samples, n_electrodes)
egm_ref = np.asarray(catheter.get('surface_ecg_sum', []), dtype=object) # shape is (n_segments, n_samples)
activation_uni = np.asarray(catheter.get('mapping_electrogram_unipolar_activation_within_woi', []), dtype=object) # shape is (n_segments, n_electrodes)

electrogram_cycle_lengths = estimate_electrogram_cycle_lengths(egm_uni_qrs_subtracted)
median_cycle_length = (
    float(np.median(electrogram_cycle_lengths))
    if electrogram_cycle_lengths.size else np.nan
)
print(
    f'Estimated median cycle length: {median_cycle_length:g} samples '
    f'from {electrogram_cycle_lengths.size} electrograms'
)

#%%
# Flatten electrode positions, then project them independently onto each mesh.
electrode_positions_all = np.vstack([
    np.asarray(segment, dtype=float).reshape(-1, 3)
    for segment in electrode_positions
])
print('Projecting electrode positions onto original and refined meshes')
electrode_positions_on_original_mesh_all, _ = utility.ui_functions.project_electrodes_to_mesh(
    np.asarray(mesh['geometry_original_vertex'], dtype=np.float64),
    np.asarray(mesh['geometry_original_face'], dtype=np.int64),
    electrode_positions_all,
)
electrode_positions_on_refined_mesh_all, _ = utility.ui_functions.project_electrodes_to_mesh(
    np.asarray(mesh['vertex'], dtype=np.float64),
    np.asarray(mesh['face'], dtype=np.int64),
    electrode_positions_all,
)

data_store = {
    'directory': directory,
    'name_prefix': name_prefix,
    'clinical_data': catheter,
    'mesh_vertex': mesh['geometry_original_vertex'],
    'mesh_face': mesh['geometry_original_face'],
    'refined_mesh_vertex': mesh['vertex'],
    'refined_mesh_face': mesh['face'],
    'segment_count': segment_count,
    'segment_electrode_count': segment_electrode_count,
    'electrode_positions': electrode_positions,
    'electrode_positions_all': electrode_positions_all,
    'electrode_positions_on_original_mesh_all': electrode_positions_on_original_mesh_all,
    'electrode_positions_on_refined_mesh_all': electrode_positions_on_refined_mesh_all,
    'egm_uni_original': egm_uni_original,
    'egm_uni_qrs_subtracted': egm_uni_qrs_subtracted,
    'egm_ref': egm_ref,
    'activation_uni': activation_uni,
    'electrogram_cycle_lengths': electrogram_cycle_lengths,
    'median_cycle_length': median_cycle_length,
    'clinical_electrogram_woi_start': int(np.asarray(catheter['clinical_electrogram_woi_start']).item()),
    'clinical_electrogram_woi_end': int(np.asarray(catheter['clinical_electrogram_woi_end']).item()),
}

save_lock = threading.Lock()


def validate_activation_updates(payload):
    """Validate a batch of edited segments without changing server state."""
    updates = payload.get('segments')
    if not isinstance(updates, list) or not updates:
        raise ValueError('segments must be a non-empty list')

    validated = []
    seen_segment_ids = set()
    woi_start = data_store['clinical_electrogram_woi_start']
    woi_end = data_store['clinical_electrogram_woi_end']

    for update_index, update in enumerate(updates):
        if not isinstance(update, dict):
            raise ValueError(f'segments[{update_index}] must be an object')

        segment_id = update.get('segment_id')
        if isinstance(segment_id, bool) or not isinstance(segment_id, int):
            raise ValueError(f'segments[{update_index}].segment_id must be an integer')
        if segment_id < 0 or segment_id >= data_store['segment_count']:
            raise ValueError(f'segment_id {segment_id} is out-of-range')
        if segment_id in seen_segment_ids:
            raise ValueError(f'segment_id {segment_id} appears more than once')
        seen_segment_ids.add(segment_id)

        raw_activation = update.get('activation_uni')
        try:
            activation = np.asarray(raw_activation, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'activation_uni for segment {segment_id} must contain only numbers'
            ) from exc

        expected_length = len(data_store['electrode_positions'][segment_id])
        if activation.ndim != 1 or activation.shape[0] != expected_length:
            raise ValueError(
                f'activation_uni for segment {segment_id} must have length {expected_length}'
            )
        if not np.all(np.isfinite(activation)):
            raise ValueError(f'activation_uni for segment {segment_id} contains a non-finite value')
        if not np.all(activation == np.trunc(activation)):
            raise ValueError(f'activation_uni for segment {segment_id} must contain integers')

        outside_woi = (activation != 0) & (
            (activation < woi_start) | (activation > woi_end)
        )
        if np.any(outside_woi):
            raise ValueError(
                f'activation_uni for segment {segment_id} must be 0 or within '
                f'the window of interest [{woi_start}, {woi_end}]'
            )

        validated.append((segment_id, activation.astype(np.int64)))

    return validated


def updated_activation_array(source, updates):
    """Return a detached activation array containing the validated updates."""
    segments = [np.asarray(source[index]).copy() for index in range(len(source))]
    for segment_id, activation in updates:
        segments[segment_id] = activation

    if isinstance(source, np.ndarray) and source.ndim > 1:
        result = np.empty(source.shape, dtype=source.dtype)
        for segment_id, activation in enumerate(segments):
            result[segment_id] = activation
        return result

    result = np.empty(len(segments), dtype=object)
    for segment_id, activation in enumerate(segments):
        result[segment_id] = activation
    return result


def atomic_savez(save_path, values):
    """Write an NPZ beside its destination and atomically replace the old file."""
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f'.{save_path.name}.', suffix='.tmp', dir=save_path.parent
    )
    try:
        if save_path.exists():
            os.fchmod(file_descriptor, save_path.stat().st_mode & 0o7777)
        with os.fdopen(file_descriptor, 'wb') as temporary_file:
            np.savez(temporary_file, **values)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, save_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def interpolate_activation_to_mesh(activation, sample_points, mesh_vertices):
    """Linearly interpolate valid projected electrode values in 3-D."""
    activation = np.asarray(activation, dtype=np.float64)
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)
    sample_points = np.asarray(sample_points, dtype=np.float64)
    valid = np.isfinite(activation) & (activation != 0) & np.all(
        np.isfinite(sample_points), axis=1
    )
    if not np.any(valid):
        return np.full(len(mesh_vertices), np.nan)

    return griddata(sample_points[valid], activation[valid], mesh_vertices, method='linear')


def interpolate_activation_phase_to_mesh(
    activation, cycle_length, sample_points, mesh_vertices
):
    """Circularly interpolate activation phase for the Full HSV colorscale."""
    activation = np.asarray(activation, dtype=np.float64)
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)
    sample_points = np.asarray(sample_points, dtype=np.float64)
    valid = np.isfinite(activation) & (activation != 0) & np.all(
        np.isfinite(sample_points), axis=1
    )
    empty = np.full(len(mesh_vertices), np.nan)
    if not np.any(valid) or not np.isfinite(cycle_length) or cycle_length <= 0:
        return empty, empty.copy(), np.nan

    phase_origin = float(np.min(activation[valid]))
    phase = np.mod((activation[valid] - phase_origin) / cycle_length, 1.0)
    angles = 2.0 * np.pi * phase
    phase_vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    mesh_vectors = griddata(
        sample_points[valid], phase_vectors, mesh_vertices, method='linear'
    )

    confidence = np.hypot(mesh_vectors[:, 0], mesh_vectors[:, 1])
    mesh_phase = np.mod(
        np.arctan2(mesh_vectors[:, 1], mesh_vectors[:, 0]) / (2.0 * np.pi),
        1.0,
    )
    ambiguous = ~np.isfinite(confidence) | (confidence <= 1e-12)
    mesh_phase[ambiguous] = np.nan
    confidence[ambiguous] = np.nan
    return mesh_phase, confidence, phase_origin

app = Flask(__name__, template_folder=directory['home'], static_folder=directory['home'], static_url_path='')
@app.route('/')
def index():
    return render_template('ui_patient_data_observer.html')

@app.route('/api/data')
def get_data():
    # Keep initial payload lightweight; electrograms are fetched on demand.
    data = {
        'name_prefix': data_store['name_prefix'],
        'mesh_vertex': data_store['mesh_vertex'].tolist(),
        'mesh_face': data_store['mesh_face'].tolist(),
        'refined_mesh_vertex': data_store['refined_mesh_vertex'].tolist(),
        'refined_mesh_face': data_store['refined_mesh_face'].tolist(),
        'electrode_positions': [np.asarray(seg, dtype=float).tolist() for seg in data_store['electrode_positions']],
        'mesh_activation': [None] * len(data_store['refined_mesh_vertex']),
        'mesh_phase': [None] * len(data_store['refined_mesh_vertex']),
        'median_cycle_length': (
            float(data_store['median_cycle_length'])
            if np.isfinite(data_store['median_cycle_length']) else None
        ),
        'cycle_length_electrogram_count': int(len(data_store['electrogram_cycle_lengths'])),
        'clinical_electrogram_woi_start': int(data_store['clinical_electrogram_woi_start']),
        'clinical_electrogram_woi_end': int(data_store['clinical_electrogram_woi_end']),
        'activation_uni': [np.asarray(seg, dtype=float).tolist() for seg in data_store['activation_uni']],
        'segment_count': int(data_store['segment_count']),
        'segment_electrode_count': int(data_store['segment_electrode_count']),
        'n_segments': int(data_store['segment_count']),
        'n_electrodes': int(data_store['segment_electrode_count'])
    }

    return jsonify(utility.ui_functions.json_safe(data))

@app.route('/api/electrograms', methods=['POST'])
def get_electrograms():
    payload = request.get_json(silent=True) or {}
    segment_id = int(payload.get('segment_id', 0))

    if segment_id < 0 or segment_id >= data_store['segment_count']:
        return jsonify({'error': 'segment_id is out-of-range'}), 400

    egm_uni = np.asarray(data_store['egm_uni_original'][segment_id], dtype=float)
    egm_uni_qrs_subtracted = np.asarray(data_store['egm_uni_qrs_subtracted'][segment_id], dtype=float)
    egm_ref = np.asarray(data_store['egm_ref'][segment_id], dtype=float)

    n_electrodes = egm_uni.shape[1] if egm_uni.ndim > 1 else 0
    if n_electrodes == 0:
        return jsonify({'error': 'no electrodes available for this segment'}), 400

    response = {
        'segment_id': segment_id,
        'electrode_ids': list(range(n_electrodes)),
        'egm_uni': [egm_uni[:, e_id].tolist() for e_id in range(n_electrodes)],
        'egm_uni_qrs_subtracted': [egm_uni_qrs_subtracted[:, e_id].tolist() for e_id in range(n_electrodes)],
        'egm_ref': [egm_ref.tolist() for _ in range(n_electrodes)],
    }
    return jsonify(utility.ui_functions.json_safe(response))


@app.route('/api/interpolate', methods=['POST'])
def interpolate_activation():
    payload = request.get_json(silent=True) or {}
    mesh_type = payload.get('mesh_type', 'refined')
    if mesh_type not in ('original', 'refined'):
        return jsonify({'error': 'mesh_type must be original or refined'}), 400
    if mesh_type == 'refined':
        mesh_vertices = data_store['refined_mesh_vertex']
        sample_points = data_store['electrode_positions_on_refined_mesh_all']
    else:
        mesh_vertices = data_store['mesh_vertex']
        sample_points = data_store['electrode_positions_on_original_mesh_all']
    # Accept either a flattened per-electrode activation array ('activation_all')
    # or a per-segment activation ('activation_uni') together with 'segment_id'.
    if 'activation_all' in payload:
        activation_all = np.asarray(payload.get('activation_all', []), dtype=np.float64)
        if activation_all.ndim != 1 or activation_all.shape[0] != data_store['electrode_positions_all'].shape[0]:
            return jsonify({'error': 'activation_all has the wrong length'}), 400
    else:
        segment_id = int(payload.get('segment_id', 0))
        activation = np.asarray(payload.get('activation_uni', []), dtype=np.float64)

        if segment_id < 0 or segment_id >= data_store['segment_count']:
            return jsonify({'error': 'segment_id is out-of-range'}), 400

        expected_shape = np.asarray(data_store['activation_uni'][segment_id], dtype=np.float64).shape
        if activation.shape != expected_shape:
            return jsonify({'error': 'activation_uni has the wrong length for the selected segment'}), 400

        activation_all = np.asarray(data_store['activation_uni'], dtype=float).copy()
        activation_all[segment_id] = activation
        activation_all = activation_all.reshape(-1)
    if payload.get('full_hsv') is True:
        cycle_length = payload.get('cycle_length', data_store['median_cycle_length'])
        if isinstance(cycle_length, bool):
            return jsonify({'error': 'cycle length must be a positive number'}), 400
        try:
            cycle_length = float(cycle_length)
        except (TypeError, ValueError):
            return jsonify({'error': 'cycle length must be a positive number'}), 400
        if not np.isfinite(cycle_length) or cycle_length <= 0:
            return jsonify({'error': 'cycle length must be a positive number'}), 422
        mesh_phase, phase_confidence, phase_origin = interpolate_activation_phase_to_mesh(
            activation_all, cycle_length, sample_points, mesh_vertices
        )
        return jsonify(utility.ui_functions.json_safe({
            'mesh_phase': [None if not np.isfinite(value) else float(value)
                           for value in mesh_phase],
            'mesh_phase_confidence': [None if not np.isfinite(value) else float(value)
                                      for value in phase_confidence],
            'phase_origin': phase_origin,
            'cycle_length': float(cycle_length),
        }))

    mesh_activation = interpolate_activation_to_mesh(
        activation_all, sample_points, mesh_vertices
    )
    return jsonify(utility.ui_functions.json_safe({
        'mesh_activation': [None if not np.isfinite(value) else float(value)
                            for value in mesh_activation]
    }))


@app.route('/api/window-of-interest', methods=['POST'])
def update_window_of_interest():
    payload = request.get_json(silent=True) or {}
    raw_woi_start = payload.get('clinical_electrogram_woi_start')
    raw_woi_end = payload.get('clinical_electrogram_woi_end')

    try:
        if isinstance(raw_woi_start, bool) or isinstance(raw_woi_end, bool):
            raise ValueError
        if isinstance(raw_woi_start, float) and not raw_woi_start.is_integer():
            raise ValueError
        if isinstance(raw_woi_end, float) and not raw_woi_end.is_integer():
            raise ValueError
        woi_start = int(raw_woi_start)
        woi_end = int(raw_woi_end)
    except (TypeError, ValueError, OverflowError):
        return jsonify({'error': 'WOI start and end must be integers'}), 400

    if woi_start < 0 or woi_start >= woi_end:
        return jsonify({'error': 'WOI must satisfy 0 <= start < end'}), 400

    electrograms = data_store['egm_uni_qrs_subtracted']
    sample_counts = [np.asarray(segment).shape[0] for segment in electrograms]
    if not sample_counts or woi_end > min(sample_counts):
        return jsonify({
            'error': f'WOI end must not exceed the electrogram length ({min(sample_counts, default=0)})'
        }), 400

    with save_lock:
        # Recompute on a detached dictionary so a failed calculation cannot leave
        # the shared catheter state partially updated.
        updated_catheter = utility.ui_functions.grab_activations_within_window_of_interest(
            dict(data_store['clinical_data']), woi_start, woi_end
        )
        activation_key = 'mapping_electrogram_unipolar_activation_within_woi'
        updated_activation = updated_catheter[activation_key]

        data_store['clinical_data']['clinical_electrogram_woi_start'] = woi_start
        data_store['clinical_data']['clinical_electrogram_woi_end'] = woi_end
        data_store['clinical_data'][activation_key] = updated_activation
        data_store['clinical_electrogram_woi_start'] = woi_start
        data_store['clinical_electrogram_woi_end'] = woi_end
        data_store['activation_uni'] = updated_activation

    return jsonify(utility.ui_functions.json_safe({
        'status': 'ok',
        'clinical_electrogram_woi_start': woi_start,
        'clinical_electrogram_woi_end': woi_end,
        'activation_uni': [np.asarray(segment, dtype=float).tolist()
                           for segment in updated_activation],
    }))


@app.route('/api/save', methods=['POST'])
def save_activation_times():
    payload = request.get_json(silent=True) or {}
    try:
        updates = validate_activation_updates(payload)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    activation_key = 'mapping_electrogram_unipolar_activation_within_woi'
    save_path = data_store['directory']['data'] / f"{data_store['name_prefix']}_catheter.npz"

    with save_lock:
        clinical_data = data_store['clinical_data']
        updated_activation = updated_activation_array(clinical_data[activation_key], updates)
        data_to_save = dict(clinical_data)
        data_to_save[activation_key] = updated_activation

        try:
            atomic_savez(save_path, data_to_save)
        except Exception:
            app.logger.exception('Failed to save activation times to %s', save_path)
            return jsonify({'error': 'Unable to write the catheter data file'}), 500

        # commit the new values to server memory only after the file replacement succeeds.
        clinical_data[activation_key] = updated_activation
        data_store['activation_uni'] = updated_activation

    saved_segment_ids = [segment_id for segment_id, _ in updates]
    print(f"Saved activation times for segments {saved_segment_ids} to {save_path}")
    return jsonify({
        'status': 'ok',
        'path': str(save_path),
        'saved_segment_ids': saved_segment_ids,
        'saved_segment_count': len(saved_segment_ids),
    })

#%%
if __name__ == '__main__':
    server_port = 5001

    # stop any stale server that is already listening on Flask's port.
    stopped_port = subprocess.run(
        ['fuser', '-k', '-TERM', f'{server_port}/tcp'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if stopped_port:
        time.sleep(5)

    # open the patient data observer user interface
    threading.Timer(1.0, webbrowser.open, args=[f'http://127.0.0.1:{server_port}']).start() # runs webbrowser.open on a background thread after a 1-second delay, while the main thread proceeds to start Flask. The 1-second delay gives Flask time to start up before the browser tries to connect
    app.run(debug=False, port=server_port, host='0.0.0.0')
