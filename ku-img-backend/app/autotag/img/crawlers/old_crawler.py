








"""
DONOT USE THIS FILE. THIS IS THE OLD CRAWLER USED JUST FOR REFERENCE.

"""







# STANDARD MODULES
import os
import time
import shutil

# IMAGE MANIPULATION
from PIL import Image, ImageEnhance, ImageFilter

# IMAGE CRAWLER
from icrawler.builtin import GoogleImageCrawler

import random
from typing import Callable, Union

separator = os.sep


# add noise to the crawled image
def add_noise(img, noise_factor=0.05):
    width, height = img.size
    num_pixels = int(noise_factor * width * height)
    for _ in range(num_pixels):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        img.putpixel((x, y), random.randint(0, 255))
    return img


# make RGB format and resize
def format_image(img):
    img = img.convert("RGB")
    img = img.resize((64, 64))
    return img


# def crawl_img(cfg, transform_func: Callable[[Image.Image], Image.Image] | None = None) :
def crawl_img(
    cfg, transform_func: Union[Callable[[Image.Image], Image.Image], None] = None
):
    """
    Crawls the internet and saves the data into the location specified in cfg

    cfg specifies the details of keywords and location
    cfg is a list of [{
        'key' : keyword,
        'max' : max no of images for keyword,
        'loc' : location where the image should be saved,
        'ext' : The extension
    })

    transform_func is an optional transformation function : f(Pillow Image Object) => Pillow Image Object
    """
    keywords = cfg.get("keywords", [])

    for elem in keywords:
        keyword = elem["key"]
        max_count = elem["max"]
        target_dir = elem["loc"]
        extension = elem["ext"]
        fcounter = 0

        # clear target if exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)

        # define crawler
        google_crawler = GoogleImageCrawler(
            parser_threads=2, downloader_threads=4, storage={"root_dir": target_dir}
        )

        # crawl
        filters = dict(size=cfg.get("size", "large"))
        google_crawler.crawl(
            keyword=keyword, filters=filters, max_num=max_count, file_idx_offset=0
        )

        # other transformations which are done for all images
        for root, dirs, files in os.walk(target_dir):
            for img_path in (os.path.join(root, _) for _ in files):
                img = Image.open(img_path)

                if (
                    transform_func is not None
                ):  # apply transform function to each image generated (if provided)
                    transformed_img = transform_func(img)

                    # delete old image
                    img.close()
                    os.remove(img_path)

                    # save new image
                    img_path = f"{img_path[: img_path.rfind(separator)]}/{fcounter:05}.{extension}"
                    transformed_img.save(img_path)

                    img = transformed_img  # all other tansformations are applied on new image

                img_path_main = os.path.join(
                    os.path.dirname(img_path), f"{fcounter:05}"
                )
                print(img_path_main)
                # generate b/w version
                img_bw = img.convert("L")
                img_bw = img_bw.resize((64, 64))
                # rotate the image
                for i in range(10):
                    angle = random.randint(0, 360)
                    img_rotated = img_bw.rotate(angle)
                    img_rotated = format_image(img_rotated)
                    rotated_img_path = f"{img_path_main}_rot_{i}.{extension}"
                    img_rotated.save(rotated_img_path)
                # noise in image
                img_noisy = add_noise(img)
                img_noisy = format_image(img_noisy)
                # blur image
                img_blurred = img_noisy.filter(ImageFilter.BLUR)
                img_blurred = format_image(img_blurred)
                # save image and it's augmentation
                img_path = f"{img_path_main}.{extension}"
                bw_img_path = f"{img_path_main}_bw.{extension}"
                noise_img_path = f"{img_path_main}_noise.{extension}"
                blurred_img_path = f"{img_path_main}_blur.{extension}"
                img_bw.save(bw_img_path)
                img_noisy.save(noise_img_path)
                img_blurred.save(blurred_img_path)
                fcounter += 1


if __name__ == "__main__":
    target_loc = os.path.join(os.getcwd(), "data", "images")
    keywords = [
        {
            "key": "arm chair",
            "max": 5,
            "loc": os.path.join(target_loc, "arm chair"),
            "ext": "jpg",
        },
        {
            "key": "bean bag",
            "max": 5,
            "loc": os.path.join(target_loc, "bean bag"),
            "ext": "jpg",
        },
        {
            "key": "bench",
            "max": 2,
            "loc": os.path.join(target_loc, "bench"),
            "ext": "jpg",
        },
        {
            "key": "gaming chair",
            "max": 1,
            "loc": os.path.join(target_loc, "gaming chair"),
            "ext": "jpg",
        },
        {
            "key": "wooden chair",
            "max": 1,
            "loc": os.path.join(target_loc, "wooden chair"),
            "ext": "jpg",
        },
    ]

    cfg = {
        "keywords": keywords,
        "size": "icon",
    }
    f = lambda img: img.convert("RGB").resize(
        (64, 64), resample=Image.Resampling.BILINEAR
    )

    crawl_img(cfg, f)
