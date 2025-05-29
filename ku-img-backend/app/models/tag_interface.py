import config
import json

# DYNAMIC IMPORTS
from importlib import import_module

# DYNAMIC INSPECTS
import inspect

# DEFAULT DICT
from collections import defaultdict
from pprint import pprint

# BASE MODELS
from models.base import MultiClassSingleTagModel

catalog_path = config.MODEL_CATALOG_PATH

def parse_catalog() : 
    model_info = {}
    tag_model_list = defaultdict(set)

    with open(catalog_path, 'r') as f : 
        entries = json.loads(f.read())
    
        for entry in entries : 
            MODULE = entry['module']
            CLASS = entry['class']
            RAW = entry['raw']
            MODEL = getattr(import_module(MODULE), CLASS)
            BASE = inspect.getmro(MODEL)[-2]	
            TAGS = MODEL.tags
    
            _model_info = {
                'module' : MODULE,
                'class' : CLASS,
                'raw' : RAW,
                'model' : MODEL,
                'base' : BASE,
                'tags' : TAGS
            }
    
            model_info[MODEL] = _model_info
    
            for tag in TAGS : 
                tag_model_list[tag].add(MODEL)

    return model_info, tag_model_list

# build the catalog structures
model_info, tag_model_list = parse_catalog()

def tag_image(loc, subset=None, include_details=False) : 
    """
        loc : location of the image
        subset : 
            subset of tags that should be considered.
            If specified, only models associated with the tags are considered
            If not specified, all models are queried for potential tags
        details : 
            return details about which model was responsible for the tag
            If True, details returns List of all models which returned that particular tag for all tags.
            If False, returns None
            
        Returns : 
            Tuple(
            List[tag(str)],
            {'tag' : List[model]} | None) (if details = True | False)
    """
    # determine models that participate in polling process

    # subset = set(['ship','horse'])
    if subset : 
        models = set()
        for _tag in subset : 
            _models = tag_model_list.get(_tag, set())
            _models = set(_models)
            models = models.union(_models)
    else : 
        models = set([v['model'] for k,v in model_info.items()])

    tags = set()
    details = defaultdict(set)
    
    for model in models : 
        try : 
            base = model_info[model]['base']

            # instantiate the model
            _model = model()

            if base == MultiClassSingleTagModel : 
                tag = _model.tag(loc)
                if tag : 
                    tags.add(tag)
                    details[tag].add(_model.__class__.__name__)

        except Exception as e : 
            print(f"Unable to query model : {model}. Encoutered : {e.__class__.__name__}:{e}")

    # subset filters
    if subset : 
        tags = [_ for _ in tags if _ in subset]
        details = {k : v for k, v in details.items() if k in subset}

    tags = sorted(list(set(tags)))
    if not include_details : details = None

    return tags, details
