import os
import json
import requests
from pprint import pprint

# paths
meta_path = os.path.join(os.getcwd(), '..', 'test_files', 'chairModel.json')
model_path = os.path.join(os.getcwd(), '..', 'test_files', 'chairModel.h5')

# parse meta
with open(meta_path, 'r') as f : 
    meta_content = f.read()
json.loads(meta_content)
meta = json.loads(meta_content)
# pprint(meta)

# read model
with open(model_path, 'rb') as f : 
    model_content = f.read()

payload = {
    'template' : 'keras/MultiClassSingleTagKerasStandardModelTemplateA.py',
    'raw_key' : 'chairModel.h5',
    'group' : 'furniture',
    'CLASS_NAME' : 'ChairClassifier_1',
    'TAGS' : str(meta['tags']),
    'MODEL_INPUT_WIDTH' : str(meta['img_dim'][0]),
    'MODEL_INPUT_HEIGHT' : str(meta['img_dim'][1]),
    'MODEL_OUTPUT_FILTERS' : "['TAG_FILTER']",
    'MODEL_TAG_FILTERS' : "['other']",
    'MODEL_THRESHOLD' : '0',
}

# pprint(payload)

response = requests.post('http://localhost:8000/model/register', files=dict(raw=model_content), data=payload)
pprint(response.json())
