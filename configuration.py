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

def map_name():
    map_id = 3

    name_prefix = [
        '103_5-2-1-1-3-Rp-ReLA CS REF 230', # 0, flutter
        '104_2-LA fam', # 1, GOOD flutter reentry
        '105_3-LA FAM', # 2, flutter
        '106_2-LA fam', # 3, flutter or focal?
        '107_3-LA CL 270', # 4, GOOD flutter reentry, best
        '109_3-LA FAM', # 5, GOOD flutter reentry, most dense
        '110_1-LA FAM', # 6, GOOD flutter reentry
        '111_6-LA', # 7, flutter or focal?
        '112_6-LA CL 300' # 8, GOOD flutter reentry
    ]

    return name_prefix[map_id]

def directory_setup():
    # directory folder
    directory = {}
    directory['home'] = script_dir
    directory['data'] = Path('/home/j/Desktop/hdd/share_folder/carto3_files/data npz') # Jay's computer
    # directory['data'] = Path('/mnt/data2/ruhi/data npz/') # Ruhi
    directory['result'] = script_dir / 'result'

    (directory['result']).mkdir(parents=True, exist_ok=True) # create the folder if it does not exist

    return directory
