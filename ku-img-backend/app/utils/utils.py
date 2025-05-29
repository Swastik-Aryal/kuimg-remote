# REDIS
import asyncio
import redis

# ZIP MANIPULATION
import zipfile
import shutil
from fastapi import Body, File, UploadFile, Request, HTTPException
import os, sys
import redis
import pathlib
import config

# ASYNC OPERATIONS
import aiofiles

fs_host, fs_port = config.REDIS_FS.split(':')
fs_conn = redis.Redis(fs_host, fs_port, socket_timeout=10)

seperator = os.sep

def zip_data(dataset_loc,key):
    try : 
        # zip all extracts and load into memory
        zip_loc = dataset_loc
        zip_loc = shutil.make_archive(zip_loc, 'zip', root_dir = dataset_loc)
        with open(zip_loc, 'rb') as f: 
            zip_content = f.read()

        # remove original zip and dataset dirs
        if os.path.exists(dataset_loc) : shutil.rmtree(dataset_loc)
        if os.path.exists(zip_loc) : os.remove(zip_loc)
       
    except Exception as e: 
        err = f'Zip operation Failed. Raised {e.__class__.__name__}:{str(e)}'
        raise HTTPException(detail = err, status_code=400)

    # Upload to Storage Server
    fs_conn.set(key, zip_content)


def save_with_other_dataset(dataset_path: pathlib.Path, new_images_path: pathlib.Path, tag: str,key: str):
    """
    Unpacks, moves and repacks into single dataset

    Parameters
    ----------
    dataset_path: The path of the dataset archive. \n
    new_images_path: The path containing the dataset of new tags (excluding others). \n
    tag: The exact name of the tag under which, the images is to be saved

    Example
    -------
    
    chair_dataset\n
    ├── chair\n
    │   ├── arm chair\n
    │   ├── gaming chair\n
    │   ├── office chair\n
    │   ├── steel chair\n
    │   └── wooden chair\n
    └── other\n
    
    which is the dataset path\n

    chair_dataset.data\n
    ├── GAN\n
    
    which is the new_images_path\n

    So, the new generated will be\n
    chair_dataset\n
    ├── chair\n
    │   ├── arm chair\n
    │   ├── gaming chair\n
    │   ├── office chair\n
    │   ├── steel chair\n
    │   └── wooden chair\n
    │   ├── GAN\n
    └── other\n

    """
    previous_working_dir = os.getcwd()

    os.chdir(config.TEMP_AUTO_IMG_PATH)

    dataset_path = dataset_path.relative_to(config.TEMP_AUTO_IMG_PATH)
    new_images_path = new_images_path.relative_to(config.TEMP_AUTO_IMG_PATH)

    print(f"dataset_path: {dataset_path.__str__()}")
    print(f"new_image_path: {new_images_path.__str__()}")
    print(f"tag: {tag}")
    
    archive_format = "zip"
    dataset_directory = dataset_path.__str__().removesuffix(f".{archive_format}")
    print(f"Dataset Directory: {dataset_directory}")

    # extract into the parent folder with same name
    if os.path.exists(dataset_directory):
        shutil.rmtree(dataset_directory.__str__())
    shutil.unpack_archive(dataset_path.absolute().__str__(), dataset_directory.__str__(), format=archive_format)

    # moving the new images into the dataset directory
    shutil.move(new_images_path.__str__(), pathlib.Path(dataset_directory).joinpath(tag).__str__())
    # deleting the parent folder containing new images
    if os.path.exists(new_images_path.parent) and len(os.listdir(new_images_path.parent)) == 0:
        os.rmdir(new_images_path.parent)

    zip_data(dataset_directory, key)
    os.chdir(previous_working_dir)


if __name__=="__main__":
    save_with_other_dataset(pathlib.Path("/home/bishal/Programming/kuimg/app/temp/auto/images/chair_dataset.zip"), pathlib.Path("/home/bishal/Programming/kuimg/app/temp/auto/images/chair_dataset.data/GAN"), "chair")
