import os
import io
import json
import requests
import shutil
import zipfile
from pprint import pprint

tag = 'jacket'
group = 'clothers'
keywords = [
    'long jacket', 
    'woolen jacket', 
    'thick jacket', 
    'leather jacket', 
]
counter_keywords = [
    'door:20', 
    'chair:20', 
    'television:30', 
    'kettle',
    'fruits',
    'animals',
]

dataset_key = f'{tag}_dataset.zip'
model_key = f'{tag}_model.zip'

# REQUEST FOR DATASET ----------------------------------------------------------
# -----------------------------------------------------------------------------
endpoint = 'http://localhost/autotag/img/fetch'
payload = {
    'tag' : tag,
    'key' : dataset_key,
    'keywords' : keywords,
    'counters' : counter_keywords
}
response = requests.post(endpoint, data=json.dumps(payload), timeout=None)
print("DONE : IMG/FETCH")


# REQUEST TO TRAIN AND RETRIEVE THE MODEL --------------------------------------
# ------------------------------------------------------------------------------
endpoint = 'http://localhost/autotag/ml/train/binary'
payload = {
    'tag' : tag,
    'dataset_key' : dataset_key,
    'key' : model_key,
}
response = requests.post(endpoint, data=json.dumps(payload), timeout=None)
pprint(response.json())
print("DONE : ML/TRAIN/BINARY")


# REQUEST TO REGISTER INTO THE API --------------------------------------------
# -----------------------------------------------------------------------------
endpoint = 'http://localhost/autotag/model/register'
payload = {
    'template' : 'keras/MultiClassSingleTagKerasStandardModelTemplateA.py',
    'group' : group,
    'model_key' : model_key
}
response = requests.post(endpoint, data=json.dumps(payload), timeout=None)
pprint(response.json())
print("DONE : MODEL/REGISTER")
