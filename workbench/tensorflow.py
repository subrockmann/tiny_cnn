import tensorflow as tf
import pandas as pd

def set_batchnorm_momentum(model, momentum=0.9):
    for layer in model.layers:
        if type(layer)==type(tf.keras.layers.BatchNormalization()):
            #print(layer.momentum)
            layer.momentum=momentum
    return model

def set_dropout(model, dropout=0.2):
    for layer in model.layers:
        if type(layer)==type(tf.keras.layers.Dropout(0)):
            #print(layer.momentum)
            layer.rate = dropout
    return model


def get_layer_details_df(filepath):
    print(f"Reading in {filepath}")
    df = pd.read_csv(filepath)
    return df