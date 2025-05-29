# STANDARD MODULES
import os
import time
import json
import shutil
import random
from datetime import datetime

# DATA PROCESSING
import numpy as np
import pandas as pd

# LOGGING 
import logging

# ML UTILS
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

# IMAGE MANIPULATION
from PIL import Image

# COLLECTIONS
from collections import defaultdict

# ZIP MANIPULATION
import zipfile

# CHARTING 
import matplotlib.pyplot as plt


def encode(tags) : 
    """ tags must be a List[Tag(str)] """
    codes, uniques = pd.factorize(tags)
    return { tag : index for index, tag in enumerate(tags) }


def parse_dataset(dataset, img_dim, test_size=0.2, random_state=None) : 
    """
        dataset = path to zipped dataset
        img_dim = (width, height) of the image in the prepared dataset
        test_size = 0-1 val indicating how to separate training and testing set
        random_state = specify a value for consistent shuffling
    """
    # param validation    
    if not random_state : random_state = random.randint(0, 100)

    # create a temporary path
    tmp_path = os.path.join(os.getcwd(), f'temp_{int(time.time())}') # change later
    shutil.rmtree(tmp_path, ignore_errors=True)
    if not os.path.exists(tmp_path) : os.makedirs(tmp_path)

    # unzip to a temporary path
    with zipfile.ZipFile(dataset, 'r') as zip_ref:
        zip_ref.extractall(tmp_path)

    # parse all root level directories as tags
    tags = os.listdir(tmp_path)

    # encode tags
    encoded_tags = encode(tags)
    print(encoded_tags)

    # prepare dataset
    # all images inside the tag folder (be it in subdirs) are considered as image for tag
    images = []
    labels = [] 
    for tag in tags : 
        tag_dataset = os.path.join(tmp_path, tag)
        for root, dirs, files in os.walk(tag_dataset) : 
            for img_path in (os.path.join(root, _) for _ in files) : 
                # transform image
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize(img_dim, resample=Image.Resampling.BILINEAR)
                img = np.asarray(img)
                img = img / 255.0

                # add to dataset
                images.append(img)
                labels.append(encoded_tags[tag])

    images = np.asarray(images).reshape(len(images), *img_dim, 3)
    labels = np.asarray(labels).reshape(len(labels), 1)

    # shuffle and return dataset 
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, shuffle=True, test_size=test_size, random_state=random_state)

    # remove temporary path
    shutil.rmtree(tmp_path, ignore_errors=True)

    # return prepared dataset 
    return (X_train, X_test, Y_train, Y_test), tags


def sample_chart(X, Y, tags) : 
    """
        X = Numpy array of Images (_, _, _, 3)
        Y = Corresponding Labels (_, 1)
        tags = List of tags where label = position of tag
    """
    plt.figure(figsize=(10, 10))
    for i in range(25) : 
        plt.subplot(5,5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])
        plt.xlabel(tags[Y[i][0]])
    return plt


def train(X_train, Y_train, img_dim, epochs) : 
  
    # add convolution layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(*img_dim, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    # add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))

    # define optimizer
    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # train
    history = model.fit(X_train, Y_train, epochs=epochs)

    return model



def generate_model(dataset, model_path, meta_path, img_dim=(64,64), epochs=25) : 
    """
        dataset must conform to the following standards : 
          - it must be a zipped file where each extracted directory is a tag
          - each directory can have images in any number of subdirectories. 
          -  However, all images will be associated with the main tag
    """  
    (X_train, X_test, Y_train, Y_test) , tags = parse_dataset(dataset, img_dim=img_dim)
    
    # # chart
    # plt = sample_chart(X_train, Y_train, tags)
    # plt.savefig('img.png')

    # train
    model = train(X_train, Y_train, img_dim=img_dim, epochs=epochs)

    # evaluate
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

    # write meta file
    meta = {
        'tags' : tags,
        'dt' : datetime.now().strftime('%Y-%m-%d'),
        'img_dim' : img_dim,
        'training_size' : len(X_train),
        'training_epochs' : epochs,
        'testing_size' : len(X_test),
        'testing_acc' : test_acc,
    }
    with open(meta_path, 'w') as f : 
        f.write(json.dumps(meta, indent=4))

    # write model_file
    model.save(model_path)
    return test_acc


if __name__ == '__main__' : 

    dataset      = os.path.join(os.environ['APP_PATH'], 'test', 'test_files', 'chairDataset.zip')
    model_output = os.path.join(os.environ['APP_PATH'], 'test', 'test_files', 'chairModel.h5')
    meta_path    = os.path.join(os.environ['APP_PATH'], 'test', 'test_files', 'chairModel.json')

    generate_model(dataset, model_output, meta_path, img_dim=[64,64], epochs=25)
