import sys
import pathlib

# resolving path problem
FILE= pathlib.Path(__file__).resolve()
ROOT=FILE.parents[1] # the app folder path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# STANDARD MODULES 
import os
import time
import json
import shutil
import requests
from pprint import pprint

# FASTAPI
from fastapi import FastAPI, Depends, UploadFile, Body, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

# TYPING
from typing import List

# STREAM RESPONSE
import io
from starlette.responses import StreamingResponse

# ASYNC FILE IO
import aiofiles

# CONFIGURATION 
import config

# DYNAMIC IMPORTS
from importlib import import_module

import glob

import models.tag_interface as mti
from models.model_gen import model_setup
import dynamic_inx.dyn_main_script as dynIdx
import dynamic_inx.model_loader



# ROUTES
from routers import autotag
from routers import fs

app = FastAPI()
app_name = config.APP_NAME

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# PING
@app.get('/')
def index() :
    html_content = """
        <html><title>{app_name}</title></head>
        <body>{app_name}@shishir</body>
        </html>
    """.format(app_name=app_name)
    return HTMLResponse(content=html_content, status_code=200)


@app.get('/status')
def status() : 
    return {str(k) : str(v) for k,v in mti.model_info.items()}
    

@app.get('/getalltags')
def alltags() :
    all_tags = list(mti.tag_model_list.keys())
    return {"data": [tag for tag in all_tags if tag != "other"]}


# @app.get('/dataFrame')
# def get_dataframe():
#     response_data= dynIdx.data_table.to_dict(orient="records")
#     return JSONResponse(content=response_data)

@app.get('/dataFrame')
def get_dataframe():
    data_frame = dynIdx.dynFunc()
    # return data_frame
    if data_frame is None:
        return "Please train models for more than 3 tags to visualize the tags embeddings,category, hierarchy."
    else:
        response_data= data_frame.to_dict(orient="records")
        return JSONResponse(content=response_data)   



@app.get('/category')
def dyn_idx():
    return dynIdx.similar_categories_dict


@app.get('/list')
def item_list():
    return dynIdx.new_all_tags()

# @app.get('/singleTag')
# def single_tag():   
#     return {str(k): str(v) for k,v in dynIdx.tag_model_list}

@app.get('/rebuild')
def rebuild() : 
    mti.model_info, mti.tag_model_list = mti.parse_catalog()
    
# Global Tagging Endpoint
@app.post('/tag')
async def tag(
    img : UploadFile = File(
        ...,
        description = 'image as form-data'
        ),
    sub : List = Body(
        None, 
        description='[Optional] subset of tags to consider.'
        ),
    include_details : bool = Body(
        False,
        description='Boolean to include/omit the details of the tag'
        )
    ) : 

    # write img into a temporary target location
    name = f'test_{int(time.time())}'
    img_loc = os.path.join(config.TEMP_DATA_PATH, name)
    async with aiofiles.open(img_loc, 'wb') as loc : 
        content = await img.read()
        await loc.write(content) 

    tags, details = mti.tag_image(loc=img_loc, subset=sub, include_details=include_details)
    if tags:
        return {'tags' : tags, 'details' : details}
    else:
        return {'tags': "We dont have model to recognise this, please train it first"}



# AUTOTAG ENDPOINTS ============================================================
app.include_router(
    autotag.router,
    prefix='/autotag',
    tags=['autotag'],
    dependencies=None
)
# ==============================================================================

# FS ENDPOINTS =================================================================
app.include_router(
    fs.router,
    prefix='/fs',
    tags=['fs'],
    dependencies=None
)
# ==============================================================================

# Registration Endpoint for new models
# The API Interface will NOT perform any action on templates.
# These templates simply act as references.
# It is the responsibility of the caller the substitue the necessary variables in the template
@app.get('/model/templates/list')
async def list_model_templates() : 
    lst = glob.glob(os.path.join(config.MODEL_TEMPLATE_PATH, '**/*.py'), recursive=True)
    lst = [ _.replace(config.MODEL_TEMPLATE_PATH, '').lstrip('/') for _ in lst if _.endswith('.py') ]
    return lst

# Register is a standard procedure which expects following parameters : 
# 1. The core model which will be placed in <MODEL_RAW_PATh>
# 2. The standard template as present in the template directory.
# 3. A json which provides substitution values for all the parameters. (Except for the default ones)
@app.post('/model/register/')
async def model_register(
    request : Request,
    raw : UploadFile = File(
        ...,
        description = 'Raw model export (.h5/...)'
        ),
    raw_key : str = Body (
        ...,
        description = 'raw model key with extension format'
        ),
    template : str = Body (
        ...,
        description = 'A standard template as returned by /model/templates/list'
        ),
    group : str = Body(
        None,
        description = 'A key to orgainize models'
        ),
    ) : 

    # extract Uppercase keys as params
    req = await request.form()
    params = {}
    for k, v in req.items() : 
        if k.upper() == k : params[k] = v

    # fetch the template
    template_path = os.path.join(config.MODEL_TEMPLATE_PATH, template)

    if not os.path.exists(template_path) : 
        raise HTTPException(detail = 'Invalid Template', status_code=400)

    # write raw into a temporary location
    name = f'{raw_key}'
    raw_loc = os.path.join(config.TEMP_RAW_PATH, name)
    async with aiofiles.open(raw_loc, 'wb') as loc : 
        content = await raw.read()
        # print(len(content))
        await loc.write(content) 

    try : 
        model_setup(raw_loc, template, params, group=group) 
        rebuild()
    except Exception as e: 
        raise HTTPException(detail = str(e), status_code=400)

    return "SUCCESS"


@app.post('/test/')
async def test(
    img : UploadFile,
    ) : 

    # write img into a temporary target location
    name = f'test_{int(time.time())}'
    img_loc = os.path.join(config.TEMP_DATA_PATH, name)
    async with aiofiles.open(img_loc, 'wb') as loc : 
        content = await img.read()
        await loc.write(content) 

    MODULE = 'models.standard.cifar10.model'
    CLASS = 'Cifar10Classifier'
    model = getattr(import_module(MODULE), CLASS)()
    return model.tag(img_loc)


@app.post('/dyntest/')
async def dyntest(
    module : UploadFile,        # For testing purpose, we will be using TestTemplate.py
    ) : 

    # move the file to a local location
    target_loc = os.path.join(config.MODEL_STANDARD_PATH, 'test.py')
    async with aiofiles.open(target_loc, 'wb') as loc : 
        content = await module.read()
        await loc.write(content)

    # add mechanism for persistence
    # import the module
    MODULE = 'models.standard.test'
    CLASS = 'Test'
    m = getattr(import_module(MODULE), CLASS)()

    # return the result
    return m.test()


@app.post('/remote_upload')
async def remote_upload(
    ) : 
    with open('requirements.txt', 'r') as f : 
        content = f.read()

    endpoint = 'http://localhost:8001/put/'
    payload = {'key' : 'remupload.txt'}
    resp = requests.post(endpoint, files=dict(dataset=content), data=payload)
    pprint(resp)

    # return content
    return "TODO"



