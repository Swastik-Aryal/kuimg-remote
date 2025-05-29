import os
import numpy as np
from PIL import Image	
from models.base import MultiClassSingleTagKerasStandardModel

# CONFIGURATION
import config
import models.tag_interface as mti

model_raw_path = config.MODEL_RAW_PATH

class Cifar10Classifier(MultiClassSingleTagKerasStandardModel) : 
    tags = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self) : 
        # extract raw path
        _raw = mti.model_info[self.__class__]['raw']
        *_raw_group, _raw_name, _raw_format = _raw.split('.')
        _raw_filename = f'{_raw_name}.{_raw_format}'
        _raw_group = '/'.join(_raw_group)
        raw_model_path = os.path.join(model_raw_path, _raw_group, _raw_filename)
        
        # instantiate
        super().__init__(raw_model_path, Cifar10Classifier.tags)
        self.__model_input_size = (32,32)


    def transform_input(self, img_path) : 
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(self.__model_input_size, resample = Image.Resampling.BILINEAR)
        img = np.asarray(img)
        img = img / 255.0
        return np.array([img,])


    def transform_output(self, model_output) : 
        threshold = 0
        arg_max = np.argmax(model_output)
        arg_max_val = model_output.flatten()[arg_max]
        if arg_max_val > threshold : 
            return self.tags[np.argmax(model_output)]
        return None
        

if __name__ == '__main__' : 
    x = Cifar10Classifier()
    img_path = '/home/shailav/kuimg/test/pillow/cat.jfif'
    print(x.tag(img_path))

