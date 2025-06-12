# STANDARD MODULES
import os
from pathlib import Path
import shutil
import json
import time
from datetime import datetime

# FASTAPI
from fastapi import Body, File, UploadFile, Request, HTTPException
import tensorflow as tf

# CUSTOM API ROUTER
from utils.fastapi.routers import CustomAPIRouter1 as APIRouter

# zipper
from utils.utils import zip_data, save_with_other_dataset

# AUTOTAG MODULE
import models.tag_interface as mti
from models.model_gen import model_setup
from autotag.img.crawlers.standard_crawler import crawl_img

# from autotag.ml.generators.standard_cnn import generate_model
from autotag.ml.generators.standard_pretrained_CNN_EarlyStop import generate_model
from autotag.ml.GAN.train import train_on_augmented_data, generate_and_save_images

# CONFIGURATION
import config

# COLLECTIONS
from collections import defaultdict

# REDIS
import redis

# ZIP MANIPULATION
import zipfile

# STREAM RESPONSE
import io
from starlette.responses import StreamingResponse

# ASYNC OPERATIONS
import aiofiles

# TYPING
from typing import List

# FS REDIS CONNECTION
fs_host, fs_port = config.REDIS_FS.split(":")
fs_conn = redis.Redis(fs_host, fs_port, socket_timeout=10)

router = APIRouter()


@router.post("/tag")
async def put(
    img: UploadFile = File(..., description="object file"),
    sub: List = Body(None, description="[Optional] subset of tags to consider."),
    include_details: bool = Body(
        False, description="Boolean to include/omit the details of the tag"
    ),
):
    """Store object to storage uniquely identified by key"""

    # write img into a temporary target location
    name = f"test_{int(time.time())}"
    img_loc = os.path.join(config.TEMP_DATA_PATH, name)
    async with aiofiles.open(img_loc, "wb") as loc:
        content = await img.read()
        await loc.write(content)

    tags, details = mti.tag_image(
        loc=img_loc, subset=sub, include_details=include_details
    )
    return {"tags": tags, "details": details}


@router.post("/img/fetch/")
def autotag_img_fetch(
    tag: str = Body(..., description="tag for which image dataset is being generated"),
    key: str = Body(
        ..., description="Output dataset is uploaded to storage server with given key"
    ),
    keywords: List = Body(
        [],
        description="list of supplementary keywords. To specify a max value use : separator. eg 'apple:50' ",
    ),
    counters: List = Body([], description="list of counter keywords to train against"),
):
    """
    Generate a dataset for the given specifications
    If target_key specified, uploads dataset to key. Else streams response
    """
    # Build arguments
    dataset_loc = os.path.join(config.TEMP_AUTO_IMG_PATH, f"{tag}.data")

    # Build crawler args for keywords
    keyword_args = defaultdict(lambda: int(config.AUTO_IMG_DFLT_KWRD_MAX))
    for _ in keywords:
        keyword, *max_specifier = _.split(":")

        if len(max_specifier) > 1:
            raise HTTPException(
                detail=f"Cannot have multiple count specifiers : {_}", status_code=400
            )
        try:
            if len(max_specifier) == 1:
                keyword_args[keyword] = int(max_specifier[0])
            else:
                keyword_args[keyword] += 0

        except Exception as e:
            # err = f'count specifier must be integer : {_}. Raised {e.__class__.__name__}:{str(e)}'
            # raise HTTPException(detail = err, status_code=400)
            raise HTTPException(
                detail=f"count specifier must be integer : {_}", status_code=400
            )

    keyword_args = [
        {
            "key": k,
            "max": v,
            "loc": os.path.join(dataset_loc, tag, k),
            "ext": "jpg",
        }
        for k, v in keyword_args.items()
    ]

    # Build crawler args for counters
    counter_args = defaultdict(lambda: int(config.AUTO_IMG_DFLT_CNTR_MAX))
    for _ in counters:
        keyword, *max_specifier = _.split(":")

        if len(max_specifier) > 1:
            raise HTTPException(
                detail=f"Cannot have multiple count specifiers : {_}", status_code=400
            )

        try:
            if len(max_specifier) == 1:
                counter_args[keyword] = int(max_specifier[0])
            else:
                counter_args[keyword] += 0
        except:
            raise HTTPException(
                detail=f"count specifier must be integer : {_}", status_code=400
            )

    counter_args = [
        {
            "key": k,
            "max": v,
            "loc": os.path.join(dataset_loc, "other", k),
            "ext": "jpg",
        }
        for k, v in counter_args.items()
    ]

    # Run crawler
    try:
        # crawl keyword images
        crawl_img({"keywords": keyword_args, "size": "icon"})

        # crawl counter images
        crawl_img(
            {
                "keywords": counter_args,
                "size": "icon",
            }
        )
    except Exception as e:
        err = f"Crawler Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=500)

    # Zip, load into memory and delete source
    # try :

    #     # zip all extracts and load into memory
    #     zip_loc = dataset_loc
    #     zip_loc = shutil.make_archive(zip_loc, 'zip', root_dir = dataset_loc)
    #     with open(zip_loc, 'rb') as f:
    #         zip_content = f.read()

    #     # remove original zip and dataset dirs
    #     if os.path.exists(dataset_loc) : shutil.rmtree(dataset_loc)
    #     if os.path.exists(zip_loc) : os.remove(zip_loc)

    # except Exception as e:
    #     err = f'Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}'
    #     raise HTTPException(detail = err, status_code=400)

    # # Upload to Storage Server
    zip_data(dataset_loc, key)

    return key


@router.post("/ml/train/binary")
async def train(
    dataset_key: str = Body(
        ...,
        description="Object Key corresponding the the zipped dataset conforming to required standards",
    ),
    key: str = Body(
        ...,
        description="Output model info is uploaded to storage server with given key",
    ),
    tag: str = Body(..., description="tag for which the dataset is provided"),
):
    """
    Train a model using a provided dataset for image classification with a specific tag.
    Parameters:
    dataset_key : str
        Object key corresponding to the zipped dataset conforming to required standards.
        The dataset should be a zip file containing properly organized images for training.
    key : str
        Output model info is uploaded to storage server with this key.
        This key is used to store and later retrieve the trained model.
    tag : str
        The tag category for which the dataset is provided.
        This defines the classification label that the model will learn to identify.
    Returns:
    tuple
        A tuple containing the following evaluation metrics:
        - test_accuracy (float): The model's accuracy on the test dataset
        - f1 (float): The F1 score of the model
        - precision (float): The precision score of the model
        - recall (float): The recall score of the model
    """

    # write dataset into a temporary location
    name = f"{dataset_key}"
    dataset_temp_loc = os.path.join(config.TEMP_AUTO_IMG_PATH, name)
    async with aiofiles.open(dataset_temp_loc, "wb") as loc:
        # content = await dataset.read()
        content = fs_conn.get(dataset_key)
        await loc.write(content)

    # specify output directory
    name = f"{tag}"
    model_dir = os.path.join(config.TEMP_AUTO_RAW_PATH, tag)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    out_model_path = os.path.join(model_dir, "model.h5")
    out_meta_path = os.path.join(model_dir, "meta.json")

    # build model
    try:
        test_accuracy, f1, precision, recall = generate_model(
            dataset_temp_loc, out_model_path, out_meta_path, img_dim=[64, 64], epochs=5
        )
    except Exception as e:
        err = f"Model Generation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Zip, load into memory and delete source
    try:
        # zip all extracts and load into memory
        zip_loc = model_dir
        zip_loc = shutil.make_archive(model_dir, "zip", root_dir=model_dir)
        with open(zip_loc, "rb") as f:
            zip_content = f.read()

        # remove original zip and dataset dirs
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        if os.path.exists(zip_loc):
            os.remove(zip_loc)
    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Upload to Storage Server
    fs_conn.set(key, zip_content)

    # return key
    return test_accuracy, f1, precision, recall


## added by shishir for GAN (Nigam Bishal code) ##


@router.post("/ml/train/gan")
async def train_with_gan(
    dataset_key: str = Body(
        ...,
        description="Object Key corresponding the the zipped dataset with GAN images conforming to required standards",
    ),
    key: str = Body(
        ...,
        description="Output model info is uploaded to storage server with given key",
    ),
    tag: str = Body(..., description="tag for which the dataset is provided"),
):
    """
    Individual GAN Training + CNN Train Endpoint
    Parameters:
    dataset_key : str
        Object key corresponding to the zipped dataset conforming to required standards.
        The dataset should be a zip file containing properly organized images for training.
    key : str
        Output model info is uploaded to storage server with this key.
        This key is used to store and later retrieve the trained model.
    tag : str
        The tag category for which the dataset is provided.
        This defines the classification label that the model will learn to identify.
    Returns:
    dict : {"accuracy": accu, "f1": f1, "precision": precision, "recall": recall}
    """
    name = f"{dataset_key}"
    dataset_key_GAN = f"{dataset_key}_GAN"
    print("/n the name variable has ", name)
    print("/n The dataset_key_GAN has this ", dataset_key_GAN)
    dataset_temp_loc = os.path.join(config.TEMP_AUTO_IMG_PATH, name)
    async with aiofiles.open(dataset_temp_loc, "wb") as loc:
        # content = await dataset.read()
        content = fs_conn.get(dataset_key)
        await loc.write(content)
    # specify output directory
    name = f"{tag}"

    images_identifier = "augmented-generic"
    # hyperparameters

    zipped_dataset_path = dataset_temp_loc
    latent_vec_dim = config.LATENT_VEC_DIMENSION
    batch_size = (
        config.GAN_BATCH_SIZE
    )  # reducing batch size becuase of lesser number of data
    n_images_to_generate = config.IMAGE_TO_GENERATE
    image_dim = (64, 64)
    # n_epoch = 400
    n_epoch = 3
    try:
        gen, _ = train_on_augmented_data(
            zipped_dataset_path=zipped_dataset_path,
            latent_dim=latent_vec_dim,
            n_epoch=n_epoch,
            batch_size=batch_size,
            image_dim=image_dim,
        )
        image_save_path = os.path.join(
            config.TEMP_AUTO_IMG_PATH, f"{name}_dataset.data"
        )
        new_images_path = os.path.join(image_save_path, "GAN")
        start_index = 0
        for i in range(0, n_images_to_generate + 1, batch_size):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, latent_vec_dim), seed=i
            )
            images = generate_and_save_images(
                gen,
                random_latent_vectors,
                new_images_path,
                images_identifier,
                start_index=start_index,
            )
            start_index = len(images)
        save_with_other_dataset(
            Path(dataset_temp_loc), Path(new_images_path), tag=name, key=dataset_key_GAN
        )
        accu, f1, precision, recall = await train(
            key=key, dataset_key=dataset_key_GAN, tag=tag
        )
    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    return {"accuracy": accu, "f1": f1, "precision": precision, "recall": recall}


## added till here (GAN)


## ADDED BY SHISHIR For Pipelining Crawl and Train ##


async def model_train(tag):
    """
    Train the dataset and return the trained model.
    Parameters:
    tag : str
    Returns:
    tuple
        A tuple containing the following evaluation metrics:
        - test_accuracy (float): The model's accuracy on the test dataset
        - f1 (float): The F1 score of the model
        - precision (float): The precision score of the model
        - recall (float): The recall score of the model

    """
    key = tag + "_model.zip"
    dataset_key = tag + "_dataset.zip"
    # write dataset into a temporary location
    name = f"{dataset_key}"
    dataset_temp_loc = os.path.join(config.TEMP_AUTO_IMG_PATH, name)
    async with aiofiles.open(dataset_temp_loc, "wb") as loc:
        # content = await dataset.read()
        content = fs_conn.get(dataset_key)
        await loc.write(content)

    # specify output directory
    name = f"{tag}"
    model_dir = os.path.join(config.TEMP_AUTO_RAW_PATH, tag)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    out_model_path = os.path.join(model_dir, "model.h5")
    out_meta_path = os.path.join(model_dir, "meta.json")

    # build model
    try:
        test_accuracy, f1, precision, recall = generate_model(
            dataset_temp_loc, out_model_path, out_meta_path, img_dim=[64, 64], epochs=25
        )
    except Exception as e:
        err = f"Model Generation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Zip, load into memory and delete source
    try:
        # zip all extracts and load into memory
        zip_loc = model_dir
        zip_loc = shutil.make_archive(model_dir, "zip", root_dir=model_dir)
        with open(zip_loc, "rb") as f:
            zip_content = f.read()

        # remove original zip and dataset dirs
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        if os.path.exists(zip_loc):
            os.remove(zip_loc)
    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Upload to Storage Server
    fs_conn.set(key, zip_content)

    return test_accuracy, f1, precision, recall


from pydantic import BaseModel


class TagRequest(BaseModel):
    tag: str


@router.post("/img/fetch_train")
# async def autotag_img_fetch_train(
#     tag : str = Body(
#         ...,
#         description = "tag for which image dataset is being generated"
#         ),
#     ) :
async def autotag_img_fetch_train(tag_request: TagRequest):
    """
    Generate a dataset for the given specifications
    If target_key specified, uploads dataset to key. Else streams response

    Parameters:
    tag_request : TagRequest ({tag: str})
    Returns:
    dict : {"accuracy": accu, "f1": f1, "precision": precision, "recall": recall}
    """
    tag = tag_request.tag
    # Build arguments
    dataset_loc = os.path.join(config.TEMP_AUTO_IMG_PATH, f"{tag}.data")

    # keywords and counters given predefined
    tag = tag
    key = tag + "_dataset.zip"
    keywords = [f"big {tag}", f"red {tag}", f"small {tag}"]
    def_counters = ["cat", "dog", "apple", "banana"]
    counters = [i for i in def_counters if i != tag]

    # Build crawler args for keywords
    keyword_args = defaultdict(lambda: int(config.AUTO_IMG_DFLT_KWRD_MAX))
    for _ in keywords:
        keyword, *max_specifier = _.split(":")

        if len(max_specifier) > 1:
            raise HTTPException(
                detail=f"Cannot have multiple count specifiers : {_}", status_code=400
            )
        try:
            if len(max_specifier) == 1:
                keyword_args[keyword] = int(max_specifier[0])
            else:
                keyword_args[keyword] += 0

        except Exception as e:
            # err = f'count specifier must be integer : {_}. Raised {e.__class__.__name__}:{str(e)}'
            # raise HTTPException(detail = err, status_code=400)
            raise HTTPException(
                detail=f"count specifier must be integer : {_}", status_code=400
            )

    keyword_args = [
        {
            "key": k,
            "max": v,
            "loc": os.path.join(dataset_loc, tag, k),
            "ext": "jpg",
        }
        for k, v in keyword_args.items()
    ]

    # Build crawler args for counters
    counter_args = defaultdict(lambda: int(config.AUTO_IMG_DFLT_CNTR_MAX))
    for _ in counters:
        keyword, *max_specifier = _.split(":")

        if len(max_specifier) > 1:
            raise HTTPException(
                detail=f"Cannot have multiple count specifiers : {_}", status_code=400
            )

        try:
            if len(max_specifier) == 1:
                counter_args[keyword] = int(max_specifier[0])
            else:
                counter_args[keyword] += 0
        except:
            raise HTTPException(
                detail=f"count specifier must be integer : {_}", status_code=400
            )

    counter_args = [
        {
            "key": k,
            "max": v,
            "loc": os.path.join(dataset_loc, "other", k),
            "ext": "jpg",
        }
        for k, v in counter_args.items()
    ]

    # Run crawler
    try:
        # crawl keyword images
        crawl_img({"keywords": keyword_args, "size": "icon"})

        # crawl counter images
        crawl_img(
            {
                "keywords": counter_args,
                "size": "icon",
            }
        )
    except Exception as e:
        err = f"Crawler Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=500)

    # Zip, load into memory and delete source
    try:
        # zip all extracts and load into memory
        zip_loc = dataset_loc
        zip_loc = shutil.make_archive(zip_loc, "zip", root_dir=dataset_loc)
        with open(zip_loc, "rb") as f:
            zip_content = f.read()

        # remove original zip and dataset dirs
        if os.path.exists(dataset_loc):
            shutil.rmtree(dataset_loc)
        if os.path.exists(zip_loc):
            os.remove(zip_loc)

    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Upload to Storage Server
    fs_conn.set(key, zip_content)

    print("The Crawling is finished now training the model")

    accu, f1, precision, recall = await model_train(tag)

    return {"accuracy": accu, "f1": f1, "precision": precision, "recall": recall}


## added by shishir till here ##


@router.post("/model/register/")
async def model_register(
    request: Request,
    model_key: str = Body(
        ..., description="Object key which contains the model data in standard format"
    ),
    template: str = Body(
        ..., description="A standard template as returned by /model/templates/list"
    ),
    group: str = Body(None, description="A key to orgainize models"),
):
    """
    Register a model using the provided model key and template.
    Parameters:
        model_key : str
            Object key which contains the model data in standard format.
        template : str
            A standard template as returned by /model/templates/list.
            This defines the structure and parameters for the model.
        group : str
            A key to organize models. If not provided, defaults to None.
    Returns:
        str : "SUCCESS"
    """
    # write zip into a temporary target location
    zip_loc = os.path.join(
        config.TEMP_DATA_PATH,
        model_key,
    )
    unzip_loc = os.path.join(config.TEMP_DATA_PATH, f"{model_key}_unzipped")
    meta_loc = os.path.join(unzip_loc, "meta.json")
    model_loc = os.path.join(unzip_loc, "model.h5")
    if not os.path.exists(unzip_loc):
        os.makedirs(unzip_loc)
    async with aiofiles.open(zip_loc, "wb") as loc:
        content = fs_conn.get(model_key)
        await loc.write(content)
    shutil.unpack_archive(zip_loc, unzip_loc)

    # read meta information
    with open(meta_loc, "r") as f:
        meta_content = json.loads(f.read())

    # determin PARAMS
    tag = meta_content["tags"][1]
    dt_sm = meta_content["dt"].replace("-", "")
    params = {
        "TAGS": str(meta_content["tags"]),
        "CLASS_NAME": f"{tag}Classifier{dt_sm}",
        "MODEL_INPUT_WIDTH": str(meta_content["img_dim"][0]),
        "MODEL_INPUT_HEIGHT": str(meta_content["img_dim"][1]),
        "MODEL_OUTPUT_FILTERS": "['TAG_FILTER']",
        "MODEL_TAG_FILTERS": "['other']",
        "MODEL_THRESHOLD": "0",
    }

    # fetch the template
    template_path = os.path.join(config.MODEL_TEMPLATE_PATH, template)

    if not os.path.exists(template_path):
        raise HTTPException(detail="Invalid Template", status_code=400)

    try:
        # register model
        model_setup(model_loc, template, params, group=group)

        # rebuild catalog
        mti.model_info, mti.tag_model_list = mti.parse_catalog()
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=400)

    return "SUCCESS"
