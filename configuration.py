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

import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

def directory_setup():
    # directory folder
    directory = {}
    directory['home'] = script_dir

    # Jay's desktop
    directory['data'] = Path('/home/j/Documents/box/carto3_files/data_npz')

    # Jay's Macbook
    # directory['data'] = Path('/Users/j/Library/CloudStorage/Box-Box/Jiyue He/ArrhythmiaNet_Project/carto3_files/data_npz')

    # Ruhi's 
    # directory['data'] = Path('/Users/ruhisamudra/Library/CloudStorage/Box-Box/ArrhythmiaNet_Project/carto3_files/data_npz')

    # Justin's
    # directory['data'] = Path('')

    directory['result'] = script_dir / 'result'

    (directory['result']).mkdir(parents=True, exist_ok=True) # create the folder if it does not exist

    return directory
