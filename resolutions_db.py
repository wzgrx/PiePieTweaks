import json
import os

# Load resolutions from JSON file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_json_path = os.path.join(_current_dir, 'js', 'resolutions.json')

with open(_json_path, 'r') as f:
    _resolutions_data = json.load(f)

RESOLUTIONS = {}
for model_type, orientations in _resolutions_data.items():
    RESOLUTIONS[model_type] = {}
    for orientation, resolutions in orientations.items():
        RESOLUTIONS[model_type][orientation] = [
            (width, height) for width, height in resolutions
        ]