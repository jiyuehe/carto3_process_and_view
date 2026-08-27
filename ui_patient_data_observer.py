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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from flask import Flask, render_template, jsonify, request
import os
import tempfile
import webbrowser
import threading
import subprocess
import time
import configuration
import utility


# Do not color mesh vertices farther than this geodesic (surface) distance from
# an electrode with a valid activation, in mesh coordinate units.
INTERPOLATION_MAX_SURFACE_DISTANCE = 10.0
SMOOTHING_SPATIAL_SIGMA = 2.5
SMOOTHING_EFFECTIVE_RADIUS = 3.0 * SMOOTHING_SPATIAL_SIGMA
SMOOTHING_HUBER_FACTOR = 1.5
SMOOTHING_DIFFUSION_STEP = 0.5


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
save_lock = threading.RLock()


def available_mesh_names():
    """Return catheter data sets that also have the mesh data required by the UI."""
    suffix = '_catheter.npz'
    names = []
    for catheter_path in directory['data'].glob(f'*{suffix}'):
        name = catheter_path.name[:-len(suffix)]
        if (directory['data'] / f'{name}_mesh.npz').is_file():
            names.append(name)
    return sorted(names, key=str.casefold)


def load_npz_values(path):
    with np.load(path, allow_pickle=True) as loaded:
        return {
            key: (loaded[key].item()
                  if isinstance(loaded[key], np.ndarray) and loaded[key].ndim == 0
                  else loaded[key])
            for key in loaded.files
        }


def prepare_surface_distance_data(vertices, faces, sample_points):
    """Prepare mesh connectivity used by coverage masking and sample smoothing."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    sample_points = np.asarray(sample_points, dtype=np.float64)

    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edge_lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
    )
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    weights = np.concatenate((edge_lengths, edge_lengths))
    graph = coo_matrix(
        (weights, (rows, columns)), shape=(len(vertices), len(vertices))
    ).tocsr()

    sample_vertex_indices = np.full(len(sample_points), -1, dtype=np.int32)
    finite_samples = np.all(np.isfinite(sample_points), axis=1)
    if np.any(finite_samples):
        sample_vertex_indices[finite_samples] = cKDTree(vertices).query(
            sample_points[finite_samples], workers=-1
        )[1]

    # An unweighted random-walk operator provides fast heat-kernel diffusion on
    # this nearly uniform triangular mesh. Choose the diffusion time so its
    # spatial standard deviation is approximately SMOOTHING_SPATIAL_SIGMA.
    adjacency = graph.copy()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_degree = np.divide(
        1.0, degree, out=np.zeros_like(degree), where=degree > 0
    )
    transition = adjacency.multiply(inverse_degree[:, None]).tocsr()
    positive_edge_lengths = edge_lengths[edge_lengths > 0]
    typical_edge_length = (
        float(np.median(positive_edge_lengths))
        if len(positive_edge_lengths) else 1.0
    )
    diffusion_time = 2.0 * (
        SMOOTHING_SPATIAL_SIGMA / typical_edge_length
    ) ** 2
    diffusion_steps = max(
        1, int(np.ceil(diffusion_time / SMOOTHING_DIFFUSION_STEP))
    )

    return {
        'graph': graph,
        'sample_vertex_indices': sample_vertex_indices,
        'smoothing_transition': transition,
        'smoothing_steps': diffusion_steps,
        'smoothing_alpha': diffusion_time / diffusion_steps,
    }


def load_mesh_data(name_prefix):
    """Load and prepare every value used by the UI for one mesh/data set."""
    mesh = load_npz_values(directory['data'] / f'{name_prefix}_mesh.npz')
    catheter = load_npz_values(directory['data'] / f'{name_prefix}_catheter.npz')

    electrode_positions = np.asarray(mesh['electrode_positions'], dtype=object)
    segment_count = len(electrode_positions)
    segment_electrode_count = (
        int(electrode_positions[0].shape[0]) if segment_count else 0
    )
    egm_uni_original = np.asarray(
        catheter.get('mapping_electrogram_unipolar', []), dtype=object
    )
    egm_uni_qrs_subtracted = np.asarray(
        catheter.get('mapping_electrogram_unipolar_qrs_subtracted', []), dtype=object
    )
    egm_ref = np.asarray(catheter.get('surface_ecg_sum', []), dtype=object)
    activation_uni = np.asarray(
        catheter.get('mapping_electrogram_unipolar_activation_within_woi', []),
        dtype=object,
    )

    electrogram_cycle_lengths = estimate_electrogram_cycle_lengths(
        egm_uni_qrs_subtracted
    )
    median_cycle_length = (
        float(np.median(electrogram_cycle_lengths))
        if electrogram_cycle_lengths.size else np.nan
    )
    print(
        f'{name_prefix}: estimated median cycle length '
        f'{median_cycle_length:g} samples from '
        f'{electrogram_cycle_lengths.size} electrograms'
    )

    electrode_positions_all = (
        np.vstack([
            np.asarray(segment, dtype=float).reshape(-1, 3)
            for segment in electrode_positions
        ])
        if segment_count else np.empty((0, 3), dtype=float)
    )
    electrode_positions_on_original_mesh_all = np.asarray(
        mesh['electrode_positions_on_original_mesh_all'], dtype=np.float64
    )
    electrode_positions_on_refined_mesh_all = np.asarray(
        mesh['electrode_positions_on_refined_mesh_all'], dtype=np.float64
    )
    original_surface_data = prepare_surface_distance_data(
        mesh['geometry_original_vertex'],
        mesh['geometry_original_face'],
        electrode_positions_on_original_mesh_all,
    )
    refined_surface_data = prepare_surface_distance_data(
        mesh['vertex'],
        mesh['face'],
        electrode_positions_on_refined_mesh_all,
    )

    return {
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
        'original_surface_data': original_surface_data,
        'refined_surface_data': refined_surface_data,
        'egm_uni_original': egm_uni_original,
        'egm_uni_qrs_subtracted': egm_uni_qrs_subtracted,
        'egm_ref': egm_ref,
        'activation_uni': activation_uni,
        'electrogram_cycle_lengths': electrogram_cycle_lengths,
        'median_cycle_length': median_cycle_length,
        'clinical_electrogram_woi_start': int(np.asarray(catheter['clinical_electrogram_woi_start']).item()),
        'clinical_electrogram_woi_end': int(np.asarray(catheter['clinical_electrogram_woi_end']).item()),
    }


mesh_names = available_mesh_names()
if not mesh_names:
    raise FileNotFoundError(
        f'No matching *_catheter.npz and *_mesh.npz files found in {directory["data"]}'
    )
initial_mesh_name = configuration.map_name()
if initial_mesh_name not in mesh_names:
    initial_mesh_name = mesh_names[0]
data_store = load_mesh_data(initial_mesh_name)


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


def surface_coverage_mask(valid_samples, surface_data):
    """Find vertices within the allowed surface distance of a valid sample."""
    sample_vertex_indices = surface_data['sample_vertex_indices']
    mesh_graph = surface_data['graph']
    source_vertices = np.unique(sample_vertex_indices[valid_samples])
    source_vertices = source_vertices[source_vertices >= 0]
    if not len(source_vertices):
        return np.zeros(mesh_graph.shape[0], dtype=bool)
    surface_distance = dijkstra(
        mesh_graph,
        directed=False,
        indices=source_vertices,
        limit=INTERPOLATION_MAX_SURFACE_DISTANCE,
        min_only=True,
    )
    return surface_distance <= INTERPOLATION_MAX_SURFACE_DISTANCE


def diffuse_mesh_values(values, surface_data):
    """Apply approximate Gaussian heat diffusion to vertex channels."""
    transition = surface_data['smoothing_transition']
    alpha = surface_data['smoothing_alpha']
    diffused = np.asarray(values, dtype=np.float64).copy()
    for _ in range(surface_data['smoothing_steps']):
        diffused += alpha * (transition @ diffused - diffused)
    return diffused


def aggregate_vertex_samples(values, vertex_indices, circular):
    """Robustly merge projected samples assigned to the same mesh vertex."""
    order = np.argsort(vertex_indices, kind='stable')
    sorted_vertices = vertex_indices[order]
    sorted_values = values[order]
    source_vertices, starts = np.unique(sorted_vertices, return_index=True)
    groups = np.split(sorted_values, starts[1:])

    if not circular:
        return source_vertices, np.asarray(
            [np.median(group) for group in groups], dtype=np.float64
        )

    # The circular medoid is a robust analogue of the median and behaves
    # correctly when phases straddle zero/the end of the cycle.
    aggregated = np.empty(len(groups), dtype=np.float64)
    for group_id, group in enumerate(groups):
        pairwise_distance = np.abs(
            np.mod(group[:, None] - group[None, :] + 0.5, 1.0) - 0.5
        )
        aggregated[group_id] = group[np.argmin(np.sum(pairwise_distance, axis=1))]
    return source_vertices, aggregated


def smooth_projected_samples(values, valid_samples, surface_data, circular=False):
    """Robust geodesic Gaussian smoothing evaluated at projected samples."""
    sample_vertex_indices = surface_data['sample_vertex_indices']
    valid_indices = np.flatnonzero(valid_samples & (sample_vertex_indices >= 0))
    smoothed = np.full(len(values), np.nan, dtype=np.float64)
    if not len(valid_indices):
        return smoothed

    source_vertices, source_values = aggregate_vertex_samples(
        np.asarray(values, dtype=np.float64)[valid_indices],
        sample_vertex_indices[valid_indices],
        circular,
    )
    vertex_count = surface_data['graph'].shape[0]
    if circular:
        source_angles = 2.0 * np.pi * source_values
        source_channels = np.column_stack(
            (np.cos(source_angles), np.sin(source_angles))
        )
    else:
        source_channels = source_values[:, None]

    # First pass estimates the local trend. Residuals from that trend provide
    # Huber weights so isolated annotation errors have limited influence.
    pilot_input = np.zeros((vertex_count, source_channels.shape[1] + 1))
    pilot_input[source_vertices, :-1] = source_channels
    pilot_input[source_vertices, -1] = 1.0
    pilot = diffuse_mesh_values(pilot_input, surface_data)
    pilot_at_sources = pilot[source_vertices, :-1] / np.maximum(
        pilot[source_vertices, -1, None], 1e-12
    )

    if circular:
        pilot_length = np.linalg.norm(pilot_at_sources, axis=1)
        pilot_unit = np.divide(
            pilot_at_sources,
            pilot_length[:, None],
            out=np.zeros_like(pilot_at_sources),
            where=pilot_length[:, None] > 1e-12,
        )
        dot = np.sum(source_channels * pilot_unit, axis=1)
        cross = (
            source_channels[:, 0] * pilot_unit[:, 1]
            - source_channels[:, 1] * pilot_unit[:, 0]
        )
        residual = np.abs(np.arctan2(cross, dot)) / (2.0 * np.pi)
        minimum_huber_delta = 0.02
    else:
        residual = np.abs(source_values - pilot_at_sources[:, 0])
        minimum_huber_delta = 2.0

    robust_scale = 1.4826 * float(np.median(residual))
    huber_delta = max(SMOOTHING_HUBER_FACTOR * robust_scale, minimum_huber_delta)
    robust_weights = np.minimum(
        1.0,
        np.divide(
            huber_delta,
            residual,
            out=np.ones_like(residual),
            where=residual > 0,
        ),
    )

    final_input = np.zeros_like(pilot_input)
    final_input[source_vertices, :-1] = source_channels * robust_weights[:, None]
    final_input[source_vertices, -1] = robust_weights
    final = diffuse_mesh_values(final_input, surface_data)
    final_at_sources = final[source_vertices, :-1] / np.maximum(
        final[source_vertices, -1, None], 1e-12
    )

    if circular:
        smoothed_source_values = np.mod(
            np.arctan2(final_at_sources[:, 1], final_at_sources[:, 0])
            / (2.0 * np.pi),
            1.0,
        )
    else:
        smoothed_source_values = final_at_sources[:, 0]

    source_lookup = np.full(vertex_count, np.nan, dtype=np.float64)
    source_lookup[source_vertices] = smoothed_source_values
    smoothed[valid_indices] = source_lookup[sample_vertex_indices[valid_indices]]
    return smoothed


def interpolate_activation_to_mesh(
    activation, sample_points, mesh_vertices, surface_data
):
    """Interpolate electrode values, masking mesh regions without nearby data."""
    activation = np.asarray(activation, dtype=np.float64)
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)
    sample_points = np.asarray(sample_points, dtype=np.float64)
    # A rejected projection is [NaN, NaN, NaN] and must not contribute.
    valid = np.isfinite(activation) & (activation != 0) & np.all(
        np.isfinite(sample_points), axis=1
    )
    if not np.any(valid):
        return np.full(len(mesh_vertices), np.nan)

    smoothed_activation = smooth_projected_samples(
        activation, valid, surface_data, circular=False
    )
    mesh_activation = griddata(
        sample_points[valid],
        smoothed_activation[valid],
        mesh_vertices,
        method='linear',
    )
    covered = surface_coverage_mask(valid, surface_data)
    mesh_activation[~covered] = np.nan
    return mesh_activation


def interpolate_activation_phase_to_mesh(
    activation,
    cycle_length,
    sample_points,
    mesh_vertices,
    surface_data,
):
    """Circularly interpolate activation phase for the Full HSV colorscale."""
    activation = np.asarray(activation, dtype=np.float64)
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)
    sample_points = np.asarray(sample_points, dtype=np.float64)
    # A rejected projection is [NaN, NaN, NaN] and must not contribute.
    valid = np.isfinite(activation) & (activation != 0) & np.all(
        np.isfinite(sample_points), axis=1
    )
    empty = np.full(len(mesh_vertices), np.nan)
    if not np.any(valid) or not np.isfinite(cycle_length) or cycle_length <= 0:
        return empty, empty.copy(), np.nan

    phase_origin = float(np.min(activation[valid]))
    phase = np.mod((activation[valid] - phase_origin) / cycle_length, 1.0)
    all_phase = np.full(len(activation), np.nan, dtype=np.float64)
    all_phase[valid] = phase
    smoothed_phase = smooth_projected_samples(
        all_phase, valid, surface_data, circular=True
    )
    angles = 2.0 * np.pi * smoothed_phase[valid]
    phase_vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    mesh_vectors = griddata(
        sample_points[valid], phase_vectors, mesh_vertices, method='linear'
    )
    mesh_vectors[
        ~surface_coverage_mask(valid, surface_data)
    ] = np.nan

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


def ui_data_payload():
    """Build the initial browser payload for the currently selected mesh."""
    return {
        'name_prefix': data_store['name_prefix'],
        'mesh_names': mesh_names,
        'mesh_vertex': data_store['mesh_vertex'].tolist(),
        'mesh_face': data_store['mesh_face'].tolist(),
        'refined_mesh_vertex': data_store['refined_mesh_vertex'].tolist(),
        'refined_mesh_face': data_store['refined_mesh_face'].tolist(),
        'electrode_positions': [
            np.asarray(segment, dtype=float).tolist()
            for segment in data_store['electrode_positions']
        ],
        'mesh_activation': [None] * len(data_store['refined_mesh_vertex']),
        'mesh_phase': [None] * len(data_store['refined_mesh_vertex']),
        'median_cycle_length': (
            float(data_store['median_cycle_length'])
            if np.isfinite(data_store['median_cycle_length']) else None
        ),
        'cycle_length_electrogram_count': int(len(data_store['electrogram_cycle_lengths'])),
        'clinical_electrogram_woi_start': int(data_store['clinical_electrogram_woi_start']),
        'clinical_electrogram_woi_end': int(data_store['clinical_electrogram_woi_end']),
        'activation_uni': [
            np.asarray(segment, dtype=float).tolist()
            for segment in data_store['activation_uni']
        ],
        'segment_count': int(data_store['segment_count']),
        'segment_electrode_count': int(data_store['segment_electrode_count']),
        'n_segments': int(data_store['segment_count']),
        'n_electrodes': int(data_store['segment_electrode_count']),
    }


@app.route('/')
def index():
    return render_template('ui_patient_data_observer.html')

@app.route('/api/data')
def get_data():
    # Keep initial payload lightweight; electrograms are fetched on demand.
    return jsonify(utility.ui_functions.json_safe(ui_data_payload()))


@app.route('/api/select-mesh', methods=['POST'])
def select_mesh():
    global data_store

    payload = request.get_json(silent=True) or {}
    name_prefix = payload.get('name_prefix')
    if not isinstance(name_prefix, str) or name_prefix not in mesh_names:
        return jsonify({'error': 'Unknown mesh name'}), 400

    if name_prefix != data_store['name_prefix']:
        try:
            # Prepare everything before replacing the shared store. Other requests
            # therefore see either the complete old data set or the complete new one.
            new_data_store = load_mesh_data(name_prefix)
        except Exception:
            app.logger.exception('Failed to load mesh data for %s', name_prefix)
            return jsonify({'error': f'Unable to load data for {name_prefix}'}), 500
        with save_lock:
            data_store = new_data_store

    return jsonify(utility.ui_functions.json_safe(ui_data_payload()))

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
        surface_data = data_store['refined_surface_data']
    else:
        mesh_vertices = data_store['mesh_vertex']
        sample_points = data_store['electrode_positions_on_original_mesh_all']
        surface_data = data_store['original_surface_data']
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
            activation_all,
            cycle_length,
            sample_points,
            mesh_vertices,
            surface_data,
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
        activation_all,
        sample_points,
        mesh_vertices,
        surface_data,
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

    with save_lock:
        if payload.get('name_prefix') != data_store['name_prefix']:
            return jsonify({
                'error': 'The selected mesh changed before this save was processed'
            }), 409
        try:
            updates = validate_activation_updates(payload)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        activation_key = 'mapping_electrogram_unipolar_activation_within_woi'
        save_path = (
            data_store['directory']['data']
            / f"{data_store['name_prefix']}_catheter.npz"
        )
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
