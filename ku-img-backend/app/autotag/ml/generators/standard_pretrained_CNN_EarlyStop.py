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
from sklearn.metrics import f1_score, precision_score, recall_score

# IMAGE MANIPULATION
from PIL import Image

# COLLECTIONS
from collections import defaultdict

# ZIP MANIPULATION
import zipfile

# CHARTING
import matplotlib.pyplot as plt


def encode(tags):
    """tags must be a List[Tag(str)]"""
    codes, uniques = pd.factorize(tags)
    return {tag: index for index, tag in enumerate(tags)}


def parse_dataset(dataset, img_dim, test_size=0.2, random_state=None):
    """
    dataset = path to zipped dataset
    img_dim = (width, height) of the image in the prepared dataset
    test_size = 0-1 val indicating how to separate training and testing set
    random_state = specify a value for consistent shuffling
    """
    # param validation
    if not random_state:
        random_state = random.randint(0, 100)

    # create a temporary path
    tmp_path = os.path.join(os.getcwd(), f"temp_{int(time.time())}")  # change later
    shutil.rmtree(tmp_path, ignore_errors=True)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    # unzip to a temporary path
    with zipfile.ZipFile(dataset, "r") as zip_ref:
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
    for tag in tags:
        tag_dataset = os.path.join(tmp_path, tag)
        for root, dirs, files in os.walk(tag_dataset):
            for img_path in (os.path.join(root, _) for _ in files):
                # transform image
                img = Image.open(img_path)
                img = img.convert("RGB")
                img = img.resize(img_dim, resample=Image.Resampling.BILINEAR)
                img = np.asarray(img)
                img = img / 255.0

                # add to dataset
                images.append(img)
                labels.append(encoded_tags[tag])

    images = np.asarray(images).reshape(len(images), *img_dim, 3)
    labels = np.asarray(labels).reshape(len(labels), 1)
    if test_size == 0.0:
        shutil.rmtree(tmp_path, ignore_errors=True)
        return (images, [], labels, []), tags
    # shuffle and return dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        images, labels, shuffle=True, test_size=test_size, random_state=random_state
    )

    # remove temporary path
    shutil.rmtree(tmp_path, ignore_errors=True)

    # return prepared dataset
    return (X_train, X_test, Y_train, Y_test), tags


def sample_chart(X, Y, tags):
    """
    X = Numpy array of Images (_, _, _, 3)
    Y = Corresponding Labels (_, 1)
    tags = List of tags where label = position of tag
    """
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])
        plt.xlabel(tags[Y[i][0]])
    return plt


def get_base_model(model_name, img_dim, num_classes):
    """
    Get the specified pre-trained model

    Args:
        model_name: One of 'efficientnet', 'resnet50', 'resnet101', 'vgg16', 'vgg19'
        img_dim: Image dimensions (width, height)
        num_classes: Number of output classes

    Returns:
        base_model: The pre-trained model without top layers
        model_info: Dictionary with model-specific information
    """
    input_shape = (*img_dim, 3)

    if model_name.lower() == "efficientnet":
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        model_info = {"name": "EfficientNetB0", "dense_units": 128, "dropout_rate": 0.2}

    elif model_name.lower() == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        model_info = {"name": "ResNet50", "dense_units": 256, "dropout_rate": 0.3}

    elif model_name.lower() == "resnet101":
        base_model = tf.keras.applications.ResNet101(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        model_info = {"name": "ResNet101", "dense_units": 256, "dropout_rate": 0.3}

    elif model_name.lower() == "vgg16":
        base_model = tf.keras.applications.VGG16(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        model_info = {"name": "VGG16", "dense_units": 64, "dropout_rate": 0.5}

    elif model_name.lower() == "vgg19":
        base_model = tf.keras.applications.VGG19(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        model_info = {"name": "VGG19", "dense_units": 64, "dropout_rate": 0.5}

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Choose from: efficientnet, resnet50, resnet101, vgg16, vgg19"
        )

    return base_model, model_info


def train(
    model_path,
    X_train,
    Y_train,
    X_test,
    Y_test,
    img_dim,
    epochs,
    model_name="efficientnet",
):
    """
    Train a model using transfer learning

    Args:
        model_path: Path to save the best model
        X_train, Y_train: Training data and labels
        X_test, Y_test: Test data and labels
        img_dim: Image dimensions
        epochs: Number of training epochs
        model_name: Pre-trained model to use ('efficientnet', 'resnet50', 'resnet101', 'vgg16', 'vgg19')
    """
    num_classes = len(np.unique(Y_train))

    # Get the base model
    base_model, model_info = get_base_model(model_name, img_dim, num_classes)

    # Freeze the base model layers
    base_model.trainable = False

    # Build the model with appropriate architecture for each base model
    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),  # Better than Flatten for most modern architectures
            layers.Dropout(model_info["dropout_rate"]),
            layers.Dense(model_info["dense_units"], activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes),  # Dynamic number of classes
        ]
    )

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print(f"Using {model_info['name']} as base model")
    print(f"Model summary:")
    model.summary()

    # Define callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        mode="max",
        verbose=1,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=0.0001, verbose=1
    )

    # Train the model
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        validation_data=(X_test, Y_test),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1,
    )

    return model, model_info


def generate_model(
    dataset,
    model_path,
    meta_path,
    img_dim=(64, 64),
    epochs=25,
    model_name="efficientnet",
):
    """
    Generate and train a model using transfer learning

    Args:
        dataset: Path to zipped dataset
        model_path: Path to save the trained model
        meta_path: Path to save model metadata
        img_dim: Image dimensions (width, height)
        epochs: Number of training epochs
        model_name: Pre-trained model to use ('efficientnet', 'resnet50', 'resnet101', 'vgg16', 'vgg19')

    Dataset must conform to the following standards:
      - it must be a zipped file where each extracted directory is a tag
      - each directory can have images in any number of subdirectories
      - However, all images will be associated with the main tag
    """
    (X_train, X_test, Y_train, Y_test), tags = parse_dataset(dataset, img_dim=img_dim)

    print(
        f"Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples"
    )
    print(f"Classes: {tags}")

    # # Uncomment to generate sample chart
    # plt = sample_chart(X_train, Y_train, tags)
    # plt.savefig('sample_images.png')
    # plt.close()

    # Train the model
    model, model_info = train(
        model_path,
        X_train,
        Y_train,
        X_test,
        Y_test,
        img_dim=img_dim,
        epochs=epochs,
        model_name=model_name,
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate metrics
    f1 = f1_score(Y_test, predicted_classes, average="weighted")
    precision = precision_score(Y_test, predicted_classes, average="weighted")
    recall = recall_score(Y_test, predicted_classes, average="weighted")

    # Create metadata
    meta = {
        "base_model": model_info["name"],
        "model_name": model_name,
        "tags": tags,
        "num_classes": len(tags),
        "dt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "img_dim": img_dim,
        "training_size": len(X_train),
        "training_epochs": epochs,
        "testing_size": len(X_test),
        "testing_acc": float(test_acc),
        "testing_loss": float(test_loss),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "model_architecture": {
            "dense_units": model_info["dense_units"],
            "dropout_rate": model_info["dropout_rate"],
        },
    }

    # Save metadata
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\nTraining completed!")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {meta_path}")

    return test_acc, f1, precision, recall


if __name__ == "__main__":
    current_path = os.getcwd()
    dataset = os.path.join(current_path, "test", "test_files", "chairDataset.zip")
    model_output = os.path.join(
        current_path, "test", "New_finetune_models", "chairModel.h5"
    )
    meta_path = os.path.join(
        current_path, "test", "New_finetune_models", "chairModel.json"
    )

    # You can now specify different models:
    # 'efficientnet' (default), 'resnet50', 'resnet101', 'vgg16', 'vgg19'
    generate_model(
        dataset=dataset,
        model_output=model_output,
        meta_path=meta_path,
        img_dim=[64, 64],
        epochs=25,
        model_name="efficientnet",  # Change this to use different models
    )
