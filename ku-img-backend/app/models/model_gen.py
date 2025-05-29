from pprint import pprint
import re
import os
import json
import config
import shutil
import time

class InvalidRequest(Exception) : pass

def substitute_template(template_path, params): 
    """
        Given template and params, substitues params to generate code

        Returns Code, Error (If Code = None)
    """
    # get template content
    with open(template_path, 'r') as f : 
        content = f.read()

    # validate that all the necessary parameters are present
    required_params = re.findall('{{[ a-zA-Z0-9_]+}}', content)
    required_params = [ _.strip('{{ }}') for _ in required_params]

    # identify missing parameters 
    missing_params = set(required_params) - set(params.keys())
    if missing_params : 
        raise InvalidRequest(f"MISSING PARAMETERS : {missing_params}")

    # get rid of extra parameters
    extra_params = set(params.keys()) - set(required_params)
    for _param in extra_params : 
        params.pop(_param, None)

    # standardize params for substitution
    std_params = {}
    for key, val in params.items() : 
        key = '{{ ' + key.strip() + ' }}'
        key = re.sub('{{[ ]*', '{{ ', key)
        key = re.sub('[ ]*}}', ' }}', key)
        std_params[key] = val

    # standardize content for substitution
    content = re.sub('{{[ ]*', '{{ ', content)
    content = re.sub('[ ]*}}', ' }}', content)

    # substitute
    for param_key, param_val in std_params.items() : 
        content = content.replace(param_key, param_val)

    return content


def model_setup(raw_model_src_path, template, params, group=None) : 
    """
	raw_model_src_path : the raw ML model that the template will refer to
        raw_model_format : the extension of the raw_model export (eg: h5)
        template : the template to use to parse the raw model 
        params : the params to substitue in template
   
        # orgainization params [optional]
        group : the hierarchy/directory in which the model will live
    """
    # VALIDATION

    if not group : group = 'default'
    try : name, *_, fmt = os.path.basename(raw_model_src_path).split('.')
    except : raise InvalidRequest("FORMAT NOT SPECIFIED")

    # group and name are directory and filenames, for safety, it must only contain [A-Za-z0-9/_/]
    # since NOT search are slightly complicated in regex, we use replace appraoch
    pattern = re.compile('[A-Za-z0-9/_]+')
    if re.sub(pattern, '', group) != '' : group = 'default'
    if re.sub(pattern, '', name) != '' : name = 'default'

    # configure paths
    name = f'{name}_{int(time.time())}'
    template_path = os.path.join(config.MODEL_TEMPLATE_PATH, template)
    model_raw_path = os.path.join(config.MODEL_RAW_PATH, group, f'{name}.{fmt}')
    model_standard_path = os.path.join(config.MODEL_STANDARD_PATH, group, f'{name}.py')

    # validate template and raw paths
    if not os.path.exists(template_path) : 
        raise InvalidRequest("INVALID TEMPLATE PATH")

    if not os.path.exists(raw_model_src_path) : 
        raise InvalidRequest("INVALID RAW MODEL PATH")

    # CODE GENERATION
    code = substitute_template(template_path, params)
    if not code : raise InvalidRequest("CODE GENERATION FAILED")

    # SETUP

    # make paths if not exists
    if not os.path.exists(os.path.dirname(model_raw_path)) : os.makedirs(os.path.dirname(model_raw_path))
    if not os.path.exists(os.path.dirname(model_standard_path)) : os.makedirs(os.path.dirname(model_standard_path))
    
    # place model in the raw path
    shutil.copy(raw_model_src_path, model_raw_path)

    # place code in the standard path
    with open(model_standard_path, 'w') as f : 
        f.write(code)

    # update catalog file
    new_module = model_standard_path.replace(config.APP_PATH, '', 1).replace('/', '.').replace('.py','').strip('.')
    new_class = params['CLASS_NAME']
    new_raw = model_raw_path.replace(config.MODEL_RAW_PATH, '', 1).replace('/', '.').strip('.')
    
    catalog_path = config.MODEL_CATALOG_PATH
    with open(catalog_path, 'r') as f : 
        entries = json.loads(f.read())
    entries.append({'module' : new_module, 'class' : new_class, 'raw' : new_raw})

    with open(catalog_path, 'w') as f : 
        f.write(json.dumps(entries, indent=4))

    return "SUCCESS"
			
if __name__ == '__main__' : 
    params = {
		'CLASS_NAME' : 'Cifar100Classifier_2',
		'TAGS' : "['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']",
                'MODEL_INPUT_WIDTH' : '32',
                'MODEL_INPUT_HEIGHT' : '32',
                'MODEL_THRESHOLD' : '0',

	}
    # model_setup('/home/shailav/kuimg/app/test/cnn_raw_tst.h5', 'keras/MultiClassSingleTagKerasStandardModelTemplateA.py', params = params, group = 'test/test2/test3_4/5')
