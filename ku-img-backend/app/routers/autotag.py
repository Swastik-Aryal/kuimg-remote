# STANDARD MODULES
import os
from pathlib import Path
import shutil
import json
import time
from datetime import datetime
import tempfile

# FASTAPI
from fastapi import Body, File, UploadFile, Request, HTTPException, status
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
from Pdf_training.Pdf_extract import extract_keywords_from_pdf

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
from typing import List, Optional

# image manipulation
from PIL import Image

# Prediction Test
import numpy as np

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


## for testing the model in bet 70 to 85


@router.post("/tagTestVerify")
async def put(
    img: UploadFile = File(..., description="object file"),
    sub: List = Body(None, description="[Optional] subset of tags to consider."),
    include_details: bool = Body(
        False, description="Boolean to include/omit the details of the tag"
    ),
    temp_tag: str = Body(..., description="Temporary tag from the front end"),
):
    """Store object to storage uniquely identified by key"""

    # write img into a temporary target location
    test_name = f"test_{int(time.time())}"
    img_loc = os.path.join(config.TEMP_DATA_PATH, test_name)
    async with aiofiles.open(img_loc, "wb") as loc:
        content = await img.read()
        await loc.write(content)

    name = f"{temp_tag}"
    print("the tag name is :", name)
    model_dir = os.path.join(config.TEMP_AUTO_RAW_PATH, temp_tag)
    # tags, details = mti.tag_image(loc=img_loc, subset=sub, include_details=include_details)

    out_model_path = os.path.join(model_dir, "model.h5")
    out_meta_path = os.path.join(model_dir, "meta.json")

    def predict_tag(img_path, model_path, meta_path, img_dim=(64, 64)):
        """
        Predict the tag for the given image using a pre-trained model.

        Parameters:
        - img_path: Path to the input image file.
        - model_path: Path to the pre-trained model file (.h5).
        - meta_path: Path to the meta file containing information about the model.
        - img_dim: Dimensions of the input image.

        Returns:
        - predictions: List of predicted tags for the input image.
        """
        # Load the meta file to get information about tags
        with open(meta_path, "r") as f:
            meta = json.load(f)
            tags = meta["tags"]

        # Load the pre-trained model
        model = tf.keras.models.load_model(model_path)

        # Load and preprocess the input image
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = img.resize(img_dim, resample=Image.Resampling.BILINEAR)
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img)

        # Convert predictions to tags based on argmax
        predicted_tags = [tags[i] for i in np.argmax(predictions, axis=1)]
        return predicted_tags

    model_test_predict = predict_tag(img_loc, out_model_path, out_meta_path)
    return {"tags": model_test_predict}


## Data Collection Crawler ENDPOINT (Page1View)
@router.post("/img/fetch/")
def autotag_img_fetch(
    tag: str = Body(..., description="tag for which image dataset is being generated"),
    key: str = Body(
        ..., description="Output dataset is uploaded to storage server with given key"
    ),
    # keywords : List = Body(
    #     [],
    #     description = "list of supplementary keywords. To specify a max value use : separator. eg 'apple:50' "
    #     ),
    # counters : List = Body(
    #     [],
    #     description = "list of counter keywords to train against"
    #     ),
):
    """
    Generate a dataset for the given specifications
    If target_key specified, uploads dataset to key. Else streams response
    """
    # Build arguments
    dataset_loc = os.path.join(config.TEMP_AUTO_IMG_PATH, f"{tag}.data")

    # keywords and counters given predefined

    keywords = [tag]
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


## Indivisual CNN Model Training Endpoint (Page2View)... keep the axios f.e call to .... .post('/autotag/ml/train/binary', data)
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
    Train the dataset and return the trained model.
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
        test_accuracy = generate_model(
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

        # # remove original zip and dataset dirs
        # if os.path.exists(model_dir) : shutil.rmtree(model_dir)
        # if os.path.exists(zip_loc) : os.remove(zip_loc)

    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Upload to Storage Server
    fs_conn.set(key, zip_content)

    # return key
    return test_accuracy


## added by shishir for GAN (Nigam Bishal code) ##


## Indivisual GAN Training + CNN Train Endpoint (Alternative to Page2View. change the f.e. call to .. post('/autotag/ml/train/gan', data) if you use page2view for gan training + CNN training)
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
    Train the dataset and return the trained model.
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
        model_key = await train(key=key, dataset_key=dataset_key_GAN, tag=tag)
    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    return model_key


## added till here (GAN)


## ADDED BY SHISHIR For Pipelining Crawl and Train CNN ##


async def model_train(tag):
    """
    Train the dataset and return the trained model.
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
        test_accuracy = generate_model(
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

        # # remove original zip and dataset dirs
        # if os.path.exists(model_dir) : shutil.rmtree(model_dir)
        # if os.path.exists(zip_loc) : os.remove(zip_loc)

    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Upload to Storage Server
    fs_conn.set(key, zip_content)

    return test_accuracy


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
    """
    tag = tag_request.tag
    # Build arguments
    dataset_loc = os.path.join(config.TEMP_AUTO_IMG_PATH, f"{tag}.data")

    # keywords and counters given predefined
    tag = tag
    key = tag + "_dataset.zip"
    keywords = [tag]
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

        # # remove original zip and dataset dirs
        # if os.path.exists(dataset_loc) : shutil.rmtree(dataset_loc)
        # if os.path.exists(zip_loc) : os.remove(zip_loc)

    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Upload to Storage Server
    fs_conn.set(key, zip_content)

    print("The Crawling is finished now training the model")

    accu = await model_train(tag)

    return accu


# Function for extacting a files from a zip file to a specified location
def extract_zip(output_dir: str, zip_file: UploadFile = File(...)):
    if not zip_file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a zip file",
        )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a temporary file to store the uploaded zip content
    temp_zip_path = f"temp_{zip_file.filename}"
    try:
        # Save uploaded file
        with open(temp_zip_path, "wb") as temp_file:
            shutil.copyfileobj(zip_file.file, temp_file)

        # Process the zip file
        extracted_count = 0
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                # Skip directory entries (they end with '/')
                if file_name.endswith("/"):
                    continue

                # Get the base filename without path
                base_name = os.path.basename(file_name)
                if not base_name:  # Skip if the basename is empty
                    continue

                # Extract the file directly to the output directory, flattening the structure
                source = zip_ref.open(file_name)
                target_path = os.path.join(output_dir, base_name)

                # Handle filename conflicts (in case files with same name exist in different subdirectories)
                if os.path.exists(target_path):
                    name, ext = os.path.splitext(base_name)
                    counter = 1
                    while os.path.exists(
                        os.path.join(output_dir, f"{name}_{counter}{ext}")
                    ):
                        counter += 1
                    target_path = os.path.join(output_dir, f"{name}_{counter}{ext}")

                with open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

                source.close()
                extracted_count += 1

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{str(e)}")

    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        # Make sure the file object is closed
        zip_file.file.close()


# Endpoint for training with user's dataset (with/wihout crawled data)
@router.post("/img/fetch_train_userdata")
async def autotag_img_fetch_train_userdata(
    tag: str = Body(
        ...,
        description="tag for which image dataset is being generated",
    ),
    dataset: UploadFile = File(..., description="User's dataset"),
    use_crawled_data: bool = Body(
        True, description="Boolean to include/omit the internet crawled data"
    ),
):
    # Build arguments
    dataset_loc = os.path.join(config.TEMP_AUTO_IMG_PATH, f"{tag}.data")

    # keywords and counters given predefined
    key = tag + "_dataset.zip"
    keywords = [tag]

    if use_crawled_data and not os.path.exists(os.path.join(dataset_loc, tag)):
        # Build crawler args for keywords
        keyword_args = defaultdict(lambda: int(config.AUTO_IMG_DFLT_KWRD_MAX))
        for _ in keywords:
            keyword, *max_specifier = _.split(":")

            if len(max_specifier) > 1:
                raise HTTPException(
                    detail=f"Cannot have multiple count specifiers : {_}",
                    status_code=400,
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
        try:
            # crawl keyword images
            crawl_img({"keywords": keyword_args, "size": "icon"})

        except Exception as e:
            err = f"Crawler Failed. Raised {e.__class__.__name__}:{str(e)}"
            raise HTTPException(detail=err, status_code=500)

        extract_zip(os.path.join(dataset_loc, tag, keywords[0]), dataset)

    elif use_crawled_data and os.path.exists(os.path.join(dataset_loc, tag)):
        print("Keyword crawled data already exists. Skipping crawling step.")
        extract_zip(os.path.join(dataset_loc, tag, keywords[0]), dataset)

    else:
        if os.path.exists(os.path.join(dataset_loc, tag)):
            shutil.rmtree(os.path.join(dataset_loc, tag))
        extract_zip(os.path.join(dataset_loc, tag, keywords[0]), dataset)

    if not os.path.exists(os.path.join(dataset_loc, "other")):
        def_counters = ["cat", "dog", "apple", "banana"]
        counters = [i for i in def_counters if i != tag]
        # Build crawler args for counters
        counter_args = defaultdict(lambda: int(config.AUTO_IMG_DFLT_CNTR_MAX))
        for _ in counters:
            keyword, *max_specifier = _.split(":")

            if len(max_specifier) > 1:
                raise HTTPException(
                    detail=f"Cannot have multiple count specifiers : {_}",
                    status_code=400,
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

        # # remove original zip and dataset dirs
        # if os.path.exists(dataset_loc) : shutil.rmtree(dataset_loc)
        # if os.path.exists(zip_loc) : os.remove(zip_loc)

    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Upload to Storage Server
    fs_conn.set(key, zip_content)

    print("The Crawling is finished now training the model")

    accu = await model_train(tag)

    return accu


## added by shishir till here For Pipelining Crawl and Train CNN ##


## added by shishir from here For Piplining GAN Train and Train CNN after low val loss (without crawl)##
class GanTagRequest(BaseModel):
    tag: str


@router.post("/ml/train/gan_cnn")
async def autotag_img_train_GAN(gan_tag_request: GanTagRequest):
    """
    Train the dataset and return the trained model.
    """
    tag = gan_tag_request.tag
    key = tag + "_model.zip"
    dataset_key = tag + "_dataset.zip"

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

        model_key = await train(key=key, dataset_key=dataset_key_GAN, tag=tag)
    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    return model_key


## added by shishir till here For Pipelining GAN Tran and Train CNN ##

## added by shishir For Pipelining CRAWL, run Gan and Train CNN ##


class FetchGanTagRequest(BaseModel):
    tag: str


@router.post("/ml/train/fetch_gan_cnn")
async def autotag_img_fetch_train_GAN(fetch_gan_tag_request: FetchGanTagRequest):
    """
    Crawl and Train the dataset and return the trained model validation accuracy.
    """
    tag = fetch_gan_tag_request.tag

    # Build arguments
    dataset_loc = os.path.join(config.TEMP_AUTO_IMG_PATH, f"{tag}.data")

    # keywords and counters given predefined
    tag = tag
    key = tag + "_dataset.zip"
    keywords = [tag]
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

        # # remove original zip and dataset dirs
        # if os.path.exists(dataset_loc) : shutil.rmtree(dataset_loc)
        # if os.path.exists(zip_loc) : os.remove(zip_loc)

    except Exception as e:
        err = f"Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}"
        raise HTTPException(detail=err, status_code=400)

    # Upload to Storage Server
    fs_conn.set(key, zip_content)

    print("The Crawling is finished now training the model")

    accu = await autotag_img_train_GAN(GanTagRequest(tag=tag))

    return accu


## added by shishir till here For Pipelining CRAWL, run Gan and Train CNN ##


@router.post("/model/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type.")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    print("Pdf temporarily saved at ", tmp_path)
    keyword_data = extract_keywords_from_pdf(tmp_path, numOfKeywords=5)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    keywords = {}
    taglist = list(mti.tag_model_list.keys())

    for each, score in keyword_data.items():
        keywords[each] = {}
        keywords[each]["score"] = score
        if each in taglist:
            keywords[each]["exists"] = True
        else:
            keywords[each]["exists"] = False

    return {"keywords": keywords}


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
