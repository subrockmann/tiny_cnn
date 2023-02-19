import os, sys, math, datetime
import pathlib
from pathlib import Path

# import workbench.config.config
from workbench.config.config import initialize
from workbench.utils.utils import create_filepaths
import re
from matplotlib import pyplot as plt
#import plotly.express as px
import pandas as pd


# Helper functions

def split_operator_col(x):
    x = str(x).strip("()")
    split_strings = x.split(";")
    #print(split_strings)
    return split_strings[-1].strip()

def get_middle_part(x):
    x = str(x).strip("()")
    split_strings = x.split("/")
    if len(split_strings) == 3:
        return split_strings[1]
    elif len(split_strings) ==4:
        return split_strings[1]
    else:
        return x
    
def clean_working_set(x):
    split_strings = x.split()
    if len(split_strings) == 2:
        return (int(split_strings[0]),int(split_strings[1]))
    elif len(split_strings) == 3:
        return (int(split_strings[0]),int(split_strings[1]),int(split_strings[2]))
    elif len(split_strings)== 4:
        return (int(split_strings[0]),int(split_strings[1]),int(split_strings[2]), int(split_strings[3]))
    elif len(split_strings)== 5:
        return (int(split_strings[0]),int(split_strings[1]),int(split_strings[2]), int(split_strings[3]), int(split_strings[4]))
    elif len(split_strings)== 6:
        return (int(split_strings[0]),int(split_strings[1]),int(split_strings[2]), int(split_strings[3]), int(split_strings[4]), int(split_strings[5]))
    elif len(split_strings)== 7:
        return (int(split_strings[0]),int(split_strings[1]),int(split_strings[2]), int(split_strings[3]), int(split_strings[4]), int(split_strings[5]), int(split_strings[6]))
        return x
    
def clean_peak_memory_df(df):
    
    df["Name"] = df["Operator"].apply(split_operator_col)
    df["Memory use"] = df["Memory use"].map(int)
    df["Name"] = df["Name"].apply(get_middle_part)
    df["Working set"] = df["Working set"].apply(clean_working_set)

    # reorder columns
    df.insert(0, "Name", df.pop("Name"))
    df.insert(3, "Operator", df.pop("Operator"))

    # rename columns
    df.columns = ["name", "tensor_IDs", "RAM_b", "operator"]
    df['index'] = df.index

    return df

def get_peak_memory_df(filepath):
    print(f"Reading in {filepath}")
    df = pd.read_csv(filepath)
    print("Cleaning up the dataframe.")
    df = clean_peak_memory_df(df)
    return df


# Tensor infos

def format_shape(x):
    x = str(x)
    x = x.strip()
    split_strings = x.split()
    joined_string = ", ".join(split_strings)
    #print(joined_string)
    joined_string = "(" + joined_string +")"
    return joined_string      

def split_operator_col(x):
    x = str(x).strip("()")
    split_strings = x.split(";")
    #print(split_strings)
    return split_strings[-1].strip()

def get_middle_part(x):
    x = str(x).strip("()")
    split_strings = x.split("/")
    if len(split_strings) == 3:
        return split_strings[1]
    elif len(split_strings) ==4:
        return split_strings[1]
    else:
        return x
    
def clean_tensor_memory_df(df):
    
    df["name"] = df["Name"].apply(split_operator_col)
    df["Size"] = df["Size"].map(int)
    df["Name"] = df["Name"].apply(get_middle_part)
    
    df["Shape"].fillna(0, inplace=True)
    df["Shape"] = df["Shape"].apply(format_shape)

    # reorder columns
    df.insert(1, "name", df.pop("name"))
    df.insert(4, "Name", df.pop("Name"))

    # rename columns
    df.columns = ["id", "name", "shape", "size_b", "name_long"]

    return df

def get_tensor_details_df(filepath):
    print(f"Reading in {filepath}")
    df = pd.read_csv(filepath)
    print("Cleaning up the dataframe.")
    df = clean_tensor_memory_df(df)
    return df
