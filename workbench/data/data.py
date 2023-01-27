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
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D,DepthwiseConv2D, MaxPooling2D, AvgPool2D, GlobalAveragePooling2D, BatchNormalization, Concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
 
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