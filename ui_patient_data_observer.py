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
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from flask import Flask, render_template, jsonify, request

import webbrowser
import threading
import subprocess
import time
import configuration


def json_safe(value):
    """Convert NumPy/NaN values into plain Python data that serializes cleanly as JSON."""
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, str)) or value is None:
        return value
    return value

# functions for projecting electrodes onto the mesh and interpolating activation times
def _closest_points_on_triangles(point, triangles):
    """Return the closest point on each triangle to a single 3-D point."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap = point - a
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    result = np.empty_like(a)
    assigned = (d1 <= 0) & (d2 <= 0)
    result[assigned] = a[assigned]

    bp = point - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    mask = (d3 >= 0) & (d4 <= d3) & ~assigned
    result[mask] = b[mask]
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~assigned
    denom = d1 - d3
    v = np.divide(d1, denom, out=np.zeros_like(d1), where=denom != 0)
    result[mask] = a[mask] + v[mask, None] * ab[mask]
    assigned |= mask

    cp = point - c
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)
    mask = (d6 >= 0) & (d5 <= d6) & ~assigned
    result[mask] = c[mask]
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~assigned
    denom = d2 - d6
    w = np.divide(d2, denom, out=np.zeros_like(d2), where=denom != 0)
    result[mask] = a[mask] + w[mask, None] * ac[mask]
    assigned |= mask

    va = d3 * d6 - d5 * d4
    mask = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~assigned
    denom = (d4 - d3) + (d5 - d6)
    w = np.divide(d4 - d3, denom, out=np.zeros_like(d3), where=denom != 0)
    result[mask] = b[mask] + w[mask, None] * (c[mask] - b[mask])
    assigned |= mask

    mask = ~assigned
    denom = va + vb + vc
    v = np.divide(vb, denom, out=np.zeros_like(vb), where=denom != 0)
    w = np.divide(vc, denom, out=np.zeros_like(vc), where=denom != 0)
    result[mask] = a[mask] + ab[mask] * v[mask, None] + ac[mask] * w[mask, None]
    return result

def _nearest_indices(query, reference, count, block_size=256):
    """Memory-bounded NumPy k-nearest-neighbour search."""
    count = min(count, len(reference))
    reference_sq = np.einsum('ij,ij->i', reference, reference)
    indices = np.empty((len(query), count), dtype=np.int32)
    distances_sq = np.empty((len(query), count), dtype=np.float64)
    for start in range(0, len(query), block_size):
        stop = min(start + block_size, len(query))
        q = query[start:stop]
        distance = (
            np.einsum('ij,ij->i', q, q)[:, None]
            + reference_sq[None, :]
            - 2.0 * q @ reference.T
        )
        np.maximum(distance, 0, out=distance)
        nearest = np.argpartition(distance, count - 1, axis=1)[:, :count]
        nearest_distance = np.take_along_axis(distance, nearest, axis=1)
        order = np.argsort(nearest_distance, axis=1)
        indices[start:stop] = np.take_along_axis(nearest, order, axis=1)
        distances_sq[start:stop] = np.take_along_axis(nearest_distance, order, axis=1)
    return indices, distances_sq

def project_electrodes_to_mesh(vertices, faces, electrodes):
    """Project each electrode onto its closest candidate mesh triangle."""
    vertex_faces = [[] for _ in range(len(vertices))]
    for face_id, face in enumerate(faces):
        for vertex_id in face:
            vertex_faces[int(vertex_id)].append(face_id)

    nearby_vertices, _ = _nearest_indices(electrodes, vertices, count=4)
    projected = np.empty_like(electrodes, dtype=np.float64)
    projection_faces = np.empty(len(electrodes), dtype=np.int32)
    triangles = vertices[faces]
    for electrode_id, vertex_ids in enumerate(nearby_vertices):
        candidates = np.unique(np.concatenate([vertex_faces[v] for v in vertex_ids]))
        points = _closest_points_on_triangles(electrodes[electrode_id], triangles[candidates])
        delta = points - electrodes[electrode_id]
        distance_sq = np.einsum('ij,ij->i', delta, delta)
        best = int(np.argmin(distance_sq))
        projected[electrode_id] = points[best]
        projection_faces[electrode_id] = candidates[best]

    return projected, projection_faces

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
segment_positions = np.asarray(catheter.get('mapping_position_unipolar', []), dtype=object)
segment_count = len(segment_positions)
segment_electrode_count = int(segment_positions[0].shape[0]) if segment_count else 0

egm_uni_original = np.asarray(catheter.get('mapping_electrogram_unipolar', []), dtype=object)
egm_bi_original = np.asarray(catheter.get('mapping_electrogram_bipolar', []), dtype=object)
egm_ref = np.asarray(catheter.get('reference_electrogram', []), dtype=object)
activation_uni = np.asarray(catheter.get('mapping_electrogram_unipolar_activation_within_woi', np.zeros((segment_count, segment_electrode_count), dtype=object)), dtype=object)
activation_bi = np.asarray(catheter.get('mapping_electrogram_bipolar_activation_within_woi', activation_uni), dtype=object)

def _flatten_segment_positions(segment_positions):
    out = []
    for seg in segment_positions:
        try:
            arr = np.asarray(seg, dtype=float)
        except Exception:
            try:
                for elem in seg:
                    e = np.asarray(elem, dtype=float)
                    if e.size == 3:
                        out.append(e.reshape(3,))
            except Exception:
                continue
        else:
            if arr.ndim == 1:
                if arr.size == 3:
                    out.append(arr.reshape(3,))
                else:
                    try:
                        for elem in seg:
                            e = np.asarray(elem, dtype=float)
                            if e.size == 3:
                                out.append(e.reshape(3,))
                    except Exception:
                        continue
            else:
                try:
                    if arr.shape[-1] == 3:
                        rows = arr.reshape(-1, 3)
                        for r in rows:
                            out.append(r)
                    else:
                        for row in arr:
                            e = np.asarray(row, dtype=float)
                            if e.size == 3:
                                out.append(e.reshape(3,))
                except Exception:
                    continue
    if len(out) == 0:
        return np.empty((0, 3), dtype=float)
    return np.vstack(out)

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
    # keep first-segment electrode positions for legacy clients
    'electrode_positions': segment_positions[0] if segment_count else np.empty((0, 3), dtype=float),
    # flattened positions for all electrodes across all segments
    'electrode_positions_all': _flatten_segment_positions(segment_positions),
    'egm_uni_original': egm_uni_original,
    'egm_bi_original': egm_bi_original,
    'egm_ref': egm_ref,
    'activation_uni': activation_uni,
    'activation_bi': activation_bi,
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
        project_electrodes_to_mesh(
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

    interpolator = RBFInterpolator(
        sample_points,
        sample_values,
        kernel='linear',
        degree=0,
        neighbors=min(32, len(sample_points)),
        smoothing=1e-10,
    )
    mesh_activation[within_threshold] = interpolator(mesh_vertices[within_threshold])
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
        'activation_bi': [np.asarray(seg, dtype=float).tolist() for seg in data_store['activation_bi']],
        'segment_count': int(data_store['segment_count']),
        'segment_electrode_count': int(data_store['segment_electrode_count']),
        'n_segments': int(data_store['segment_count']),
        'n_electrodes': int(data_store['segment_electrode_count'])
    }

    return jsonify(json_safe(data))

@app.route('/api/electrograms', methods=['POST'])
def get_electrograms():
    payload = request.get_json(silent=True) or {}
    segment_id = int(payload.get('segment_id', 0))

    if segment_id < 0 or segment_id >= data_store['segment_count']:
        return jsonify({'error': 'segment_id is out-of-range'}), 400

    egm_uni = np.asarray(data_store['egm_uni_original'][segment_id], dtype=float)
    egm_bi = np.asarray(data_store['egm_bi_original'][segment_id], dtype=float) if data_store['segment_count'] > 0 and data_store['egm_bi_original'].size > 0 else np.empty((egm_uni.shape[0], 0), dtype=float)
    egm_ref = np.asarray(data_store['egm_ref'][segment_id], dtype=float)

    n_electrodes = egm_uni.shape[1] if egm_uni.ndim > 1 else 0
    if n_electrodes == 0:
        return jsonify({'error': 'no electrodes available for this segment'}), 400

    response = {
        'segment_id': segment_id,
        'electrode_ids': list(range(n_electrodes)),
        'egm_uni': [egm_uni[:, e_id].tolist() for e_id in range(n_electrodes)],
        'egm_bi': [egm_bi[:, min(e_id, egm_bi.shape[1] - 1)].tolist() if egm_bi.shape[1] > 0 else [] for e_id in range(n_electrodes)],
        'egm_ref': [egm_ref.tolist() for _ in range(n_electrodes)],
    }
    return jsonify(json_safe(response))


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

        # Expand segment activation into the flattened full electrode array
        total = data_store['electrode_positions_all'].shape[0]
        activation_all = np.zeros((total,), dtype=np.float64)
        # determine start index of this segment in flattened ordering
        start = 0
        for s in range(segment_id):
            start += int(np.asarray(data_store['segment_positions'][s]).shape[0])
        activation_all[start:start + activation.shape[0]] = activation
        mesh_activation = interpolate_activation_to_mesh(activation_all)
    return jsonify(json_safe({
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
