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
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from flask import Flask, render_template, jsonify, request
import webbrowser
import threading
import subprocess
import time
import configuration
import utility

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
segment_positions = np.asarray(catheter.get('mapping_position_unipolar', []), dtype=object) # shape is (n_segments,)
segment_count = len(segment_positions)
segment_electrode_count = int(segment_positions[0].shape[0])
egm_uni_original = np.asarray(catheter.get('mapping_electrogram_unipolar', []), dtype=object) # shape is (n_segments, n_samples, n_electrodes)
egm_uni_qrs_subtracted = np.asarray(catheter.get('mapping_electrogram_unipolar_qrs_subtracted', []), dtype=object) # shape is (n_segments, n_samples, n_electrodes)
egm_ref = np.asarray(catheter.get('reference_electrogram', []), dtype=object) # shape is (n_segments, n_samples)
activation_uni = np.asarray(catheter.get('mapping_electrogram_unipolar_activation_within_woi', []), dtype=object) # shape is (n_segments, n_electrodes)

#%%
# grab all electrode positions across all segments
flattened = []
for seg in segment_positions:
    arr = np.asarray(seg, dtype=float).reshape(-1, 3)
    flattened.append(arr)
electrode_positions_all = np.vstack(flattened)

data_store = {
    'directory': directory,
    'name_prefix': name_prefix,
    'clinical_data': catheter,
    'node_positions': mesh['geometry_original_vertex'],
    'mesh_vertex': mesh['geometry_original_vertex'],
    'mesh_face': mesh['geometry_original_face'],
    'mesh_edge': mesh['geometry_original_edge'],
    'segment_positions': segment_positions,
    'segment_count': segment_count,
    'segment_electrode_count': segment_electrode_count,
    'electrode_positions': segment_positions[0],
    'electrode_positions_all': electrode_positions_all,
    'egm_uni_original': egm_uni_original,
    'egm_uni_qrs_subtracted': egm_uni_qrs_subtracted,
    'egm_ref': egm_ref,
    'activation_uni': activation_uni,
    'clinical_electrogram_woi_start': int(np.asarray(catheter['clinical_electrogram_woi_start']).item()),
    'clinical_electrogram_woi_end': int(np.asarray(catheter['clinical_electrogram_woi_end']).item()),
}

# Geometry-dependent projection is cached because it does not change when
# activation times are edited.
interpolation_cache = directory['result'] / f'{name_prefix}_mesh_interpolation.npz'
try:
    cached = np.load(interpolation_cache)
    if (int(cached['algorithm_version']) != 2 or
            cached['mesh_shape'].tolist() != list(data_store['mesh_vertex'].shape) or
            cached['electrode_shape'].tolist() != list(data_store['electrode_positions_all'].shape)):
        raise ValueError('stale interpolation cache')
    projected_electrodes = cached['projected_electrodes']
    projection_faces = cached['projection_faces']
except (OSError, KeyError, ValueError):
    projected_electrodes, projection_faces = (
        utility.ui_functions.project_electrodes_to_mesh(
            data_store['mesh_vertex'], data_store['mesh_face'], data_store['electrode_positions_all']
        )
    )
    np.savez_compressed(
        interpolation_cache,
        algorithm_version=2,
        mesh_shape=data_store['mesh_vertex'].shape,
        electrode_shape=data_store['electrode_positions_all'].shape,
        projected_electrodes=projected_electrodes,
        projection_faces=projection_faces,
    )

data_store.update({
    'projected_electrodes': projected_electrodes,
    'projection_faces': projection_faces,
})

INTERPOLATION_DISTANCE_MM = 10.0


def _build_linear_interpolator(sample_points, sample_values):
    """Return a true linear interpolator with a local affine extrapolation fallback."""
    sample_points = np.asarray(sample_points, dtype=np.float64)
    sample_values = np.asarray(sample_values, dtype=np.float64)

    def predictor(query_points):
        query_points = np.asarray(query_points, dtype=np.float64)
        if query_points.ndim == 1:
            query_points = query_points[None, :]
        if len(query_points) == 0:
            return np.empty((0,), dtype=np.float64)
        if len(sample_points) == 1:
            return np.full(len(query_points), sample_values[0], dtype=np.float64)

        interpolator = LinearNDInterpolator(sample_points, sample_values, fill_value=np.nan)
        predicted = interpolator(query_points)
        nan_mask = ~np.isfinite(predicted)
        if not np.any(nan_mask):
            return predicted

        tree = cKDTree(sample_points)
        k = min(8, len(sample_points))
        _, neighbor_indices = tree.query(query_points[nan_mask], k=k)
        neighbor_indices = np.asarray(neighbor_indices, dtype=int)

        for row_idx, q in enumerate(query_points[nan_mask]):
            idx = neighbor_indices[row_idx]
            local_points = sample_points[idx]
            local_values = sample_values[idx]
            design = np.hstack([local_points, np.ones((len(local_points), 1))])
            coeffs, *_ = np.linalg.lstsq(design, local_values, rcond=None)
            predicted[np.flatnonzero(nan_mask)[row_idx]] = np.dot(np.hstack([q, 1.0]), coeffs)
        return predicted

    return predictor


def interpolate_activation_to_mesh(activation):
    """Interpolate within INTERPOLATION_DISTANCE_MM of valid projected electrodes; leave the rest gray."""
    activation = np.asarray(activation, dtype=np.float64)
    valid = np.isfinite(activation) & (activation != 0)
    if not np.any(valid):
        return np.full(len(data_store['mesh_vertex']), np.nan)

    sample_points = data_store['projected_electrodes'][valid]
    sample_values = activation[valid]

    # Coincident projected locations make the local interpolation system singular.
    # Combine them using their mean activation before fitting.
    unique_points, inverse = np.unique(sample_points, axis=0, return_inverse=True)
    if len(unique_points) != len(sample_points):
        sums = np.bincount(inverse, weights=sample_values)
        counts = np.bincount(inverse)
        sample_values = sums / counts
        sample_points = unique_points

    mesh_vertices = data_store['mesh_vertex']
    nearest_distance, _ = cKDTree(sample_points).query(mesh_vertices, k=1)
    within_threshold = nearest_distance <= INTERPOLATION_DISTANCE_MM
    mesh_activation = np.full(len(mesh_vertices), np.nan)
    if not np.any(within_threshold):
        return mesh_activation

    if len(sample_points) == 1:
        mesh_activation[within_threshold] = sample_values[0]
        return mesh_activation

    interpolator = _build_linear_interpolator(sample_points, sample_values)
    predicted = interpolator(mesh_vertices[within_threshold])
    mesh_activation[within_threshold] = predicted
    return mesh_activation

app = Flask(__name__, template_folder=directory['home'], static_folder=directory['home'], static_url_path='')
@app.route('/')
def index():
    return render_template('ui_patient_data_observer.html')

@app.route('/api/data')
def get_data():
    # Keep initial payload lightweight; electrograms are fetched on demand.
    data = {
        'name_prefix': data_store['name_prefix'],
        'node_positions': data_store['node_positions'].tolist(),
        'mesh_vertex': data_store['mesh_vertex'].tolist(),
        'mesh_face': data_store['mesh_face'].tolist(),
        'mesh_edge': data_store['mesh_edge'].tolist(),
        'segment_positions': [np.asarray(seg, dtype=float).tolist() for seg in data_store['segment_positions']],
        'electrode_positions': np.asarray(data_store['electrode_positions'], dtype=float).tolist(),
        'mesh_activation': [None] * len(data_store['mesh_vertex']),
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
    # Accept either a flattened per-electrode activation array ('activation_all')
    # or a per-segment activation ('activation_uni') together with 'segment_id'.
    if 'activation_all' in payload:
        activation_all = np.asarray(payload.get('activation_all', []), dtype=np.float64)
        if activation_all.ndim != 1 or activation_all.shape[0] != data_store['electrode_positions_all'].shape[0]:
            return jsonify({'error': 'activation_all has the wrong length'}), 400
        mesh_activation = interpolate_activation_to_mesh(activation_all)
    else:
        segment_id = int(payload.get('segment_id', 0))
        activation = np.asarray(payload.get('activation_uni', []), dtype=np.float64)

        if segment_id < 0 or segment_id >= data_store['segment_count']:
            return jsonify({'error': 'segment_id is out-of-range'}), 400

        expected_shape = np.asarray(data_store['activation_uni'][segment_id], dtype=np.float64).shape
        if activation.shape != expected_shape:
            return jsonify({'error': 'activation_uni has the wrong length for the selected segment'}), 400

        activation_all = np.asarray(data_store['activation_uni'], dtype=float).reshape(-1)
        mesh_activation = interpolate_activation_to_mesh(activation_all)
    return jsonify(utility.ui_functions.json_safe({
        'mesh_activation': [None if not np.isfinite(value) else float(value)
                            for value in mesh_activation]
    }))

@app.route('/api/save', methods=['POST'])
def save_activation_times():
    payload = request.get_json(silent=True) or {}
    segment_id = int(payload.get('segment_id', 0))
    activation_uni = payload.get('activation_uni')

    if segment_id < 0 or segment_id >= data_store['segment_count']:
        return jsonify({'error': 'segment_id is out-of-range'}), 400

    activation_uni = np.asarray(activation_uni, dtype=int)
    save_path = data_store['directory']['data'] / f"{data_store['name_prefix']}_clinical.npz"

    clinical_data = data_store['clinical_data']
    clinical_data['mapping_electrogram_unipolar_activation_within_woi'][segment_id] = activation_uni
    np.savez(save_path, **clinical_data)
    print(f"Saved updated activation times for segment {segment_id} to {save_path}")

    return jsonify({'status': 'ok', 'path': str(save_path), 'segment_id': segment_id})

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
        time.sleep(0.5)

    # open the patient data observer user interface
    threading.Timer(1.0, webbrowser.open, args=[f'http://127.0.0.1:{server_port}']).start() # runs webbrowser.open on a background thread after a 1-second delay, while the main thread proceeds to start Flask. The 1-second delay gives Flask time to start up before the browser tries to connect
    app.run(debug=False, port=server_port, host='0.0.0.0')
