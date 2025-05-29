# STANDARD MODULES
import io
import os
import re
import glob
import shutil

# FASTAPI
from fastapi import File, UploadFile, Query, Body, Path
from fastapi import HTTPException

# STREAM RESPONSE
import io
from starlette.responses import StreamingResponse

# TYPING
from typing import Union, List

# CUSTOM API ROUTER
from utils.fastapi.routers import CustomAPIRouter1 as APIRouter

# REDIS
import redis

# CONFIGURATION
import config

# ASYNC OPERATIONS
import aiofiles

# COMPRESSION
import gzip

# FS REDIS CONNECTION
fs_host, fs_port = config.REDIS_FS.split(':')
fs_conn = redis.Redis(fs_host, fs_port, socket_timeout=10)

router = APIRouter()

@router.post("/put")
async def put(
    raw : UploadFile = File(
        ...,
        description = 'object file'
        ),
    key : str = Body(
        ...,
        description = 'unique object key'
        ),
    ):
    """ Store object to storage uniquely identified by key """

    # extract object content 
    content = await raw.read()

    # encode
    fs_conn.set(key, content)
    return key


@router.get("/list/")
def list(
    pattern : Union[str, None] = Query(
        '*',
        description = 'Filter parameter to only fetch keys that match the pattern'
        ),
    ):
    return fs_conn.keys(pattern=pattern)


@router.get("/get")
def get_key(
    key : str = Query(
        title = 'Object Key'
        ),
    ) : 
    """ Retrieve the sepcified object """
    content = fs_conn.get(key)
    if content : 
        response = StreamingResponse(io.BytesIO(content))
        response.headers["Content-Disposition"] = f"attachment; filename={key}"
        return response
    else : 
        raise HTTPException(detail = f'NotFound:{key}', status_code=404)
    

@router.post("/delete/")
def delete_key(
    keys : List[str] = Body(
        ...,
        description = 'List of Object Key to delete',
        ),
    ):
    """ Deletes the specified object """
    fs_conn.delete(*keys)
    return keys
