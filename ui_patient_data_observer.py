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
from flask import Flask, render_template, jsonify, request

import configuration

#%%
# setting
directory = configuration.directory_setup()
name_prefix = configuration.map_name()

# load map data
data = np.load(directory['data'] / f'{name_prefix}_clinical.npz', allow_pickle=True)
clinical_data = {
    k: (data[k].item() if isinstance(data[k], np.ndarray) and data[k].ndim == 0 else data[k])
    for k in data.files
}

# variable to store data
data_store = {
    'directory': directory,
    'name_prefix': name_prefix,
    'clinical_data': clinical_data,
    'node_positions': clinical_data['geometry_original_vertex'],
    'mesh_vertex': clinical_data['geometry_original_vertex'],
    'mesh_face': clinical_data['geometry_original_face'],
    'mesh_dege': clinical_data['geometry_original_edge'],
    'electrode_positions': clinical_data['electrode_positions'],
    'egm_uni_original': clinical_data['clinical_electrogram_unipolar_original'],
    'egm_bi_original': clinical_data['clinical_electrogram_bipolar_original'],
    'egm_ref': clinical_data['clinical_electrogram_reference'],
    'activation_uni': clinical_data['clinical_activation_uni'],
    'activation_bi': clinical_data['clinical_activation_bi'],
}

app = Flask(__name__, template_folder=script_dir, static_folder=script_dir, static_url_path='')
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
        'mesh_dege': data_store['mesh_dege'].tolist(),
        'electrode_positions': data_store['electrode_positions'].tolist(),
        'clinical_electrogram_woi_start': int(data_store['clinical_data']['clinical_electrogram_woi_start']),
        'clinical_electrogram_woi_end': int(data_store['clinical_data']['clinical_electrogram_woi_end']),
        'activation_uni': data_store['activation_uni'].tolist(),
        'activation_bi': data_store['activation_bi'].tolist(),
        'n_electrodes': len(data_store['electrode_positions'])
    }
        
    return jsonify(data)

@app.route('/api/electrograms', methods=['POST'])
def get_electrograms():
    payload = request.get_json(silent=True) or {}
    electrode_ids = payload.get('electrode_ids') or []

    try:
        electrode_ids = [int(e_id) for e_id in electrode_ids]
    except (TypeError, ValueError):
        return jsonify({'error': 'electrode_ids must be a list of integers'}), 400

    n_electrodes = len(data_store['electrode_positions'])
    if any((e_id < 0 or e_id >= n_electrodes) for e_id in electrode_ids):
        return jsonify({'error': 'electrode_ids contains out-of-range index'}), 400

    egm_uni = data_store['egm_uni_original']
    egm_bi = data_store['egm_bi_original']
    egm_ref = data_store['egm_ref']

    response = {
        'electrode_ids': electrode_ids,
        'egm_uni': [egm_uni[e_id].tolist() for e_id in electrode_ids],
        'egm_bi': [egm_bi[e_id].tolist() for e_id in electrode_ids],
        'egm_ref': [egm_ref[e_id].tolist() for e_id in electrode_ids],
    }
    return jsonify(response)

@app.route('/api/save', methods=['POST'])
def save_activation_times():
    payload = request.get_json(silent=True) or {}
    activation_uni = payload.get('activation_uni')

    activation_uni = np.asarray(activation_uni, dtype=int)

    save_path = data_store['directory']['data'] / f"{data_store['name_prefix']}_clinical.npz"

    clinical_data = data_store['clinical_data']
    clinical_data['clinical_activation_uni'] = activation_uni

    np.savez(save_path, **clinical_data)
    print(f"Saved updated activation times to {save_path}")

    return jsonify({'status': 'ok', 'path': str(save_path)})

#%%
if __name__ == '__main__':
    # open the patient data observer user interface
    import webbrowser
    import threading

    threading.Timer(1.0, webbrowser.open, args=['http://127.0.0.1:5000']).start() # runs webbrowser.open on a background thread after a 1-second delay, while the main thread proceeds to start Flask. The 1-second delay gives Flask time to start up before the browser tries to connect
    app.run(debug=False, port=5000, host='0.0.0.0')
