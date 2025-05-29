import os
import numpy as np
from PIL import Image	
from models.base import MultiClassSingleTagKerasStandardModel

# CONFIGURATION
import config
import models.tag_interface as mti

model_raw_path = config.MODEL_RAW_PATH

class {{ CLASS_NAME }} (MultiClassSingleTagKerasStandardModel) : 
    tags = list({{ TAGS }})			# List of supported tags

    def __init__(self) : 
        # extract raw path
        _raw = mti.model_info[self.__class__]['raw']
        *_raw_group, _raw_name, _raw_format = _raw.split('.')
        _raw_filename = f'{_raw_name}.{_raw_format}'
        _raw_group = '/'.join(_raw_group)
        raw_model_path = os.path.join(model_raw_path, _raw_group, _raw_filename)
        
        # instantiate
        super().__init__(raw_model_path, {{ CLASS_NAME }}.tags)
        self.__model_input_size = ( {{ MODEL_INPUT_WIDTH }}, {{ MODEL_INPUT_HEIGHT }} )


    def transform_input(self, img_path) : 
        # Standard Input processing approach A
        # Image in img_path is parsed as a PIL object and transformed based on given input size.
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(self.__model_input_size, resample = Image.Resampling.BILINEAR)
        img = np.asarray(img)
        img = img / 255.0
        return np.array([img,])


    def transform_output(self, model_output) : 
        # Standard Output processing approaches 
        # Model output is considered to be an array of confidence/ probabilities

        filters = list({{ MODEL_OUTPUT_FILTERS }})
        tag_filters = list({{ MODEL_TAG_FILTERS }})
        threshold = float({{ MODEL_THRESHOLD }})

        arg_max = np.argmax(model_output)
        arg_max_val = model_output.flatten()[arg_max]
        tag = self.tags[arg_max]
        
        # FILTER LOGIC 
        if 'TAG_FILTER' in filters and tag in tag_filters : return None
        if 'THRESHOLD_FILTER' in filters and arg_max_val < threshold: return None

        return tag
