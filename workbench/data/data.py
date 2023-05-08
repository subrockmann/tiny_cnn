import os, sys, math, datetime, io
import pathlib
from pathlib import Path
import numpy as np
import random
from matplotlib import pyplot as plt
import PIL
import PIL.Image
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
#from tensorflow import keras
keras = tf.keras
from keras.layers import Input, Dense, Flatten, Conv2D,DepthwiseConv2D, MaxPooling2D, AvgPool2D, GlobalAveragePooling2D, BatchNormalization, Concatenate
from keras.layers import ReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
 
# Import the necessary MLTK APIs
from mltk.core import view_model, summarize_model, profile_model

# import workbench.config.config
from workbench.config.config import initialize
from workbench.utils.utils import create_filepaths


def get_vvw_dataset(input_shape, batch_size):
    
    vvw_path = Path.cwd().parent.joinpath("person_detection","datavisualwakewords")

    train_filenames = [str(p) for p in vvw_path.glob(f"train.record*")]
    train_tfrecords = tf.data.TFRecordDataset(train_filenames)

    val_filenames = [str(p) for p in vvw_path.glob(f"val.record*")]

    
    val_tfrecords = tf.data.TFRecordDataset(val_filenames[:6])
    test_tfrecords = tf.data.TFRecordDataset(val_filenames[6:])

    def _map_fn(example):
        return _example_to_tensors(example, input_shape)
    
    def _example_to_tensors(example, input_shape):
        """
        @brief: Read a serialized tf.train.Example and convert it to a (image, label) pair of tensors.
                TFRecords are created using src/create_coco_vww_tf_record.py 
        @author: Daniel Tan
        """
        example = tf.io.parse_example(
            example[tf.newaxis], {
                'image/encoded': tf.io.FixedLenFeature(shape = (), dtype=tf.string),
                'image/class/label': tf.io.FixedLenFeature(shape = (), dtype=tf.int64)
            })
        img_tensor =  tf.io.decode_jpeg(example['image/encoded'][0], channels = 3)
        img_tensor = tf.image.resize(img_tensor, size=(input_shape[0], input_shape[1]))
        #img_tensor = tf.expand_dims(img_tensor, axis=0)
        label = example['image/class/label']
        return img_tensor, label
    
    train_ds = train_tfrecords.map(_map_fn)
    val_ds = val_tfrecords.map(_map_fn)
    test_ds = test_tfrecords.map(_map_fn)

    # This split requires to much RAM for processing!
    # val_ds, test_ds = tf.keras.utils.split_dataset(remainder_ds, left_size=0.8, shuffle=False, seed=1)

    normalization_layer = tf.keras.layers.Rescaling(1./255, offset=-1)
    train_ds= train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds= val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds= test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(1)

    labels = ["no_person", "person"]

    return train_ds, val_ds, test_ds, labels


def get_lemon_quality_dataset(dataset_path, img_width, img_height, batch_size, channels, normalize=True):
    """ Fetches the lemon quality dataset and prints dataset info. It normalizes the image data to range [0,1] by default.

    Args: 
        dataset_path (Path): the file location of the dataset. Subfolders "train", "test", and "val" are expected.
        normalize (boolean): Normalizes the image data to range [0, 1]. Default: True

    Returns:
        (train_ds, val_ds, test_ds, class_names) (tuple(tf.datasets)): Tensorflow datasets for train, validation and test.
    
    """

    shuffle_seed = 1
    if dataset_path.exists():
        try:
            train_dir = dataset_path.joinpath("train")
            val_dir = dataset_path.joinpath( "val")
            test_dir = dataset_path.joinpath( "test")
        except:
            print(f"Please check the folder structure of {dataset_path}.")
            raise

    channels = int(channels) #.strip("c"))
    if channels==1:
        color_mode = "grayscale"
    else:
        color_mode = "rgb" 
    print(f"Color mode: {color_mode}")

    # create the labels list to avoid inclusion of .ipynb checkpoints
    #labels = ["bad_quality", "empty_background", "good_quality"]

    print("Preparing training dataset...")        
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        subset=None,
        seed=shuffle_seed,
        image_size=((img_height, img_width)),
        #labels=labels,
        batch_size=batch_size,
        color_mode=color_mode,
        shuffle=True
        )
    

    class_names = train_ds.class_names


    print("Preparing validation dataset...")    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        subset=None,
        seed=shuffle_seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode=color_mode,
        shuffle=True
        )
    

    print("Preparing test dataset...")    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        subset=None,
        seed=shuffle_seed,
        image_size=(img_height, img_width),
        batch_size=1,
        color_mode=color_mode,
        shuffle=False
        )
    
    # Create a data augmentation stage with horizontal flipping, rotations, zooms
    data_augmentation = keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
        )

    #train_ds= train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE )

    
    # Normalize the data to the range [0, 1]
    if normalize:
        normalization_layer = tf.keras.layers.Rescaling(1./255, offset=-1)

        train_ds= train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds= val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        test_ds= test_ds.map(lambda x, y: (normalization_layer(x), y)) #, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        pass

    print (f"Class names: {class_names}")
    print(f"Train: {train_ds.element_spec}")
    print(f"Normalize: {normalize}")
    return (train_ds, val_ds, test_ds, class_names)


def get_lemon_binary_datagen(dataset_path, img_width, img_height, batch_size, channels, normalize=True):
    lemon_binary_dataset_path = Path.cwd().joinpath("datasets", "lemon_dataset_binary")
    TRAIN_DIR = lemon_binary_dataset_path.joinpath("train")
    VAL_DIR = lemon_binary_dataset_path.joinpath("val")
    TEST_DIR = lemon_binary_dataset_path.joinpath("test")
    #BASE_DIR_TEST = Path.cwd().parent.joinpath("tiny_mlperf", "vw_coco2014_96_test")
    #Path.exists(BASE_DIR)
    validation_split = 0
    color_mode = "rgb"

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        #validation_split=validation_split,
        rescale=1. / 255)
    
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        #subset='training',
        color_mode=color_mode,
        class_mode="sparse")
    

    valgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    val_generator = valgen.flow_from_directory(
        VAL_DIR,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        #subset='validation',
        color_mode=color_mode,
        class_mode="sparse")
    
    test_gen =  tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    
    test_generator = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=(img_height, img_width),
        batch_size=batch_size, # was 1
        #subset = "validation",
        color_mode=color_mode,
        class_mode="sparse")
    
    #print (f"Class names: {class_names}")
    #print(f"Train: {train_generator.element_spec}")
    #print(f"Normalize: {normalize}")

    class_names  = ["bad_quality", "good_quality"]
    return (train_generator, val_generator, test_generator, class_names)



def get_vvw_minval_dataset(img_width, img_height, batch_size, channels, normalize=True):
    
    BASE_DIR = Path.cwd().parent.joinpath("tiny_mlperf", "vw_coco2014_96")
    BASE_DIR_TEST = Path.cwd().parent.joinpath("tiny_mlperf", "vw_coco2014_96_test")
    Path.exists(BASE_DIR)
    validation_split = 0.1
    color_mode = "rgb"
    shuffle_seed = 1


    print("Preparing vvw_minval_training dataset...")        
    train_ds = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR,
        validation_split=validation_split,
        subset="training",
        seed=shuffle_seed,
        image_size=(img_height, img_width),
        #labels=labels,
        batch_size=batch_size,
        color_mode=color_mode,
        shuffle=True
        )

    class_names = train_ds.class_names

    print("Preparing vvw_minval_validation dataset...")        
    val_ds = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR,
        validation_split=validation_split,
        subset="validation",
        seed=shuffle_seed,
        image_size=(img_height, img_width),
        #labels=labels,
        batch_size=batch_size,
        color_mode=color_mode,
        shuffle=True
        )

    print("Preparing test dataset...")        
    test_ds = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR_TEST,
        validation_split=None,
        #subset="validation",
        seed=shuffle_seed,
        image_size=(img_height, img_width),
        #labels=labels,
        batch_size=1,
        color_mode=color_mode,
        shuffle=False
        )


    # Create a data augmentation stage with horizontal flipping, rotations, zooms
    data_augmentation = keras.Sequential([
            tf.keras.layers.RandomRotation(10), #0.1
            tf.keras.layers.RandomTranslation(
                height_factor = 0.05,
                width_factor = 0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomFlip("horizontal")
        ]
        )

    train_ds= train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE )
    val_ds= val_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE )

    # Normalize the data to the range [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255, offset=-1)

    train_ds= train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds= val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds= test_ds.map(lambda x, y: (normalization_layer(x), y)) #, num_parallel_calls=tf.data.AUTOTUNE)
    labels = class_names
    
    print (f"Class names: {class_names}")
    print(f"Train: {train_ds.element_spec}")
    print(f"Normalize: {normalize}")
    return (train_ds, val_ds, test_ds, class_names)
