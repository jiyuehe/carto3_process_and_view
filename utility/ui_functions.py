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

import numpy as np
from scipy.spatial import cKDTree

# find the closest point on each triangle to a single 3-D point
def closest_points_on_triangles(point, triangles):
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

# k-nearest-neighbour search backed by a spatial index
def nearest_indices(query, reference, count, block_size=256):
    del block_size  # Retained for compatibility with existing callers.
    count = min(count, len(reference))
    if count == 0:
        return (
            np.empty((len(query), 0), dtype=np.int32),
            np.empty((len(query), 0), dtype=np.float64),
        )

    distances, indices = cKDTree(reference).query(query, k=count, workers=-1)
    # scipy squeezes the neighbour dimension when k == 1.
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return indices.astype(np.int32, copy=False), np.square(distances)

def mesh_vertex_normals(vertices, faces):
    """Return area-weighted unit vertex normals for a triangle mesh."""
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)

    lengths = np.linalg.norm(vertex_normals, axis=1)
    usable = np.isfinite(lengths) & (lengths > 0)
    vertex_normals[usable] /= lengths[usable, None]
    vertex_normals[~usable] = np.nan
    return vertex_normals


def project_electrodes_to_mesh(
    vertices,
    faces,
    electrodes,
    normal_vicinity=1.0,
    projection_max_distance=10.0,
    normal_candidate_count=16,
):
    """Project electrodes near a vertex-normal line; reject the rest with NaNs.

    Normal lines are treated as bidirectional so the result does not depend on
    the mesh's face-winding direction.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    electrodes = np.asarray(electrodes, dtype=np.float64)
    if normal_vicinity < 0:
        raise ValueError('normal_vicinity must be non-negative')
    if projection_max_distance < 0:
        raise ValueError('projection_max_distance must be non-negative')

    # Project each electrode onto its closest candidate mesh triangle
    vertex_faces = [[] for _ in range(len(vertices))]
    for face_id, face in enumerate(faces):
        for vertex_id in face:
            vertex_faces[int(vertex_id)].append(face_id)

    vertex_normals = mesh_vertex_normals(vertices, faces)
    nearby_vertices, _ = nearest_indices(
        electrodes, vertices, count=max(4, normal_candidate_count)
    )
    nearby_offsets = electrodes[:, None, :] - vertices[nearby_vertices]
    nearby_normals = vertex_normals[nearby_vertices]
    axial_offsets = np.einsum('evi,evi->ev', nearby_offsets, nearby_normals)
    perpendicular_offsets = (
        nearby_offsets - axial_offsets[:, :, None] * nearby_normals
    )
    distance_to_normal = np.linalg.norm(perpendicular_offsets, axis=2)
    normal_matches = (
        (distance_to_normal <= normal_vicinity)
        & (np.abs(axial_offsets) <= projection_max_distance)
    )

    projected = np.full_like(electrodes, np.nan, dtype=np.float64)
    projection_faces = np.full(len(electrodes), -1, dtype=np.int32)
    triangles = vertices[faces]
    for electrode_id, vertex_ids in enumerate(nearby_vertices):
        matched_vertex_ids = vertex_ids[normal_matches[electrode_id]]
        if not len(matched_vertex_ids):
            continue
        candidate_face_groups = [
            vertex_faces[v] for v in matched_vertex_ids if vertex_faces[v]
        ]
        if not candidate_face_groups:
            continue
        candidates = np.unique(np.concatenate(candidate_face_groups))
        points = closest_points_on_triangles(electrodes[electrode_id], triangles[candidates])
        delta = points - electrodes[electrode_id]
        distance_sq = np.einsum('ij,ij->i', delta, delta)
        best = int(np.argmin(distance_sq))
        projected[electrode_id] = points[best]
        projection_faces[electrode_id] = candidates[best]

    return projected, projection_faces

# convert NumPy/NaN values into plain Python data that serializes cleanly as JSON
def json_safe(value):
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

# grab activations within window of interest
def grab_activations_within_window_of_interest(catheter,t_start,t_end):
    catheter['clinical_electrogram_woi_start'] = t_start
    catheter['clinical_electrogram_woi_end'] = t_end

    mapping_electrogram_unipolar_activation = catheter['mapping_electrogram_unipolar_activation']
    n_segment = len(mapping_electrogram_unipolar_activation)
    _, n_channels = catheter['mapping_electrogram_unipolar_qrs_subtracted'][0].shape

    mapping_electrogram_unipolar_activation_within_woi = [None for _ in range(n_segment)]
    activation_time_edge_buffer = 10 # remove activation near the two ends of the window of interest
    for n in range(n_segment):
        activation_times_unipolar_refined = mapping_electrogram_unipolar_activation[n]
        if activation_times_unipolar_refined is None:
            activation_times_unipolar_refined = [np.array([], dtype=int) for _ in range(n_channels)]

        activation_times_within_woi = np.zeros((len(activation_times_unipolar_refined),), dtype=object)
        for channel_idx in range(len(activation_times_unipolar_refined)):
            activation_times = activation_times_unipolar_refined[channel_idx]
            if activation_times is not None and np.ptp(catheter['mapping_electrogram_unipolar_qrs_subtracted'][n][t_start:t_end,channel_idx]) >= 0.1:
                temp = activation_times[(activation_times >= t_start+activation_time_edge_buffer) & (activation_times <= t_end-activation_time_edge_buffer)]
                if len(temp) != 0:
                    activation_times_within_woi[channel_idx] = temp[0]

        mapping_electrogram_unipolar_activation_within_woi[n] = activation_times_within_woi

    catheter['mapping_electrogram_unipolar_activation_within_woi'] = mapping_electrogram_unipolar_activation_within_woi

    return catheter
