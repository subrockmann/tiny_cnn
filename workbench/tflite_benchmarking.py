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
def clean_model_summary(filepath): 

    if not filepath.is_file():
        print(f"This file does not exist: {filepath}")
        return None

    else:
        clean_lines = []
        # Parse the MLTK model summary to grab important metrics   
        with open(filepath, "r", encoding="latin-1") as f:
            lines = f.readlines() # list containing lines of file
            for line in lines:
                line = line.strip() # remove leading/trailing white spaces
                if len(line)> 0:
                    
                    clean_lines.append(line)
                else:
                    pass
            #columns = [] # To store column names
        return clean_lines


def clean_column_names(df):
    cols = df.columns

    clean_cols = []
    for col in cols:
        col = col.strip()
        col = col.replace("[" , "")
        col = col.replace("]" , "")   
        clean_cols.append(col)
        
    return clean_cols

def string_percent_to_float(x):
    x = str(x).strip("%")
    return float(x)


def remove_tabs(text):
    """removes tabs from a list of strings

    Args:
        text (list(str)): list of strings that contains tabs

    Returns:
        list(str): list of strings without tabs
    """
    split_text= []
    for l in text:
        split_text.append((l.split("\t")))
        
    return split_text

# profiling functions

def get_profiling_lines(lines):
    #print(lines)
    # find section split
    for i, line in enumerate(lines):
        #print(f"{i} : {line}")
    #if line.str.contains('= Run Order ='):
        if "Operator-wise Profiling " in line:
            split_line = i
            #print(f"Splitting text at line {split_line}")
            model_profiling = lines[:split_line]
            operator_profiling = lines[split_line+1:]

        else:
            pass
    if i==0:     
        print("WARNING: Could not find the text: 'Operator-wise Profiling Info for Regular Benchmark Runs:'")
        return (None, None)
    else:
        return model_profiling, operator_profiling

def get_index_of_model_profiling_lines(lines):
    model_lines_dict = {}
    for i, line in enumerate(lines):
        if "= Run Order =" in line:
            model_lines_dict["run_order"] = i
        elif "= Top by Computation Time =" in line:
            model_lines_dict["top_by_computation_time"] = i
        elif "The input model file size" in line:
            model_lines_dict["model_file_size"] = i
        elif "Initialized session in" in line:
            model_lines_dict["initialization_ms"] = i
        elif "Inference timings in us:" in line:
            model_lines_dict["inference_timings"] = i

    #print(model_lines_dict)
    return model_lines_dict

def get_model_profiling_stats(model_profiling):
    model_lines_dict = get_index_of_model_profiling_lines(model_profiling)

    model_stats = {}

    # extract model size
    model_size_string = model_profiling[model_lines_dict["model_file_size"]]
    model_size_string
    model_size= float(model_size_string.split(":")[-1].strip())
    model_stats["model_size_MB"] = model_size

    # extract inference times
    inference_string = model_profiling[model_lines_dict["inference_timings"]]
    #print(inference_string)
    inference_strings = inference_string.split(",")
    for item in inference_strings:
        item_time = item.split()[-1]
        #print(item_time)

    model_stats["init_us"] = int(inference_strings[0].split()[-1].strip())
    model_stats["first_inference_us"] = int(inference_strings[1].split()[-1].strip())
    model_stats["warmup_avg_us"] = float(inference_strings[2].split()[-1].strip())
    model_stats["inference_avg_us"] = float(inference_strings[3].split()[-1].strip())

    # model initialization timings
    init_string = model_profiling[model_lines_dict["initialization_ms"]]
    model_stats["initialization_ms"] = float(init_string.split()[-1].replace("ms.",""))


    # initialization run order
    init_run_order = model_profiling[model_lines_dict["run_order"]: model_lines_dict["top_by_computation_time"]]
    init_run_order = remove_tabs(init_run_order)

    model_stats["modify_graph_with_delegate_ms_first"] = float(init_run_order[2][1].strip())
    model_stats["modify_graph_with_delegate_ms_avg"] = float(init_run_order[2][2].strip())
    model_stats["modify_graph_with_delegate_ms_%"] = float(init_run_order[2][3].strip().replace("%", ""))
    model_stats["modify_graph_with_delegate_mem_KB"] = float(init_run_order[2][5].strip())

    model_stats["allocate_tensors_ms_first"] = float(init_run_order[3][1].strip())
    model_stats["allocate_tensors_ms_avg"] = float(init_run_order[3][2].strip())
    model_stats["allocate_tensors_ms_%"] = float(init_run_order[3][3].strip().replace("%", ""))

    return model_stats

def get_profiling_stats_cpu(filepath):
    """
    Parse "bechmarking.txt" file that has been generated from a .tflite model with TensorFlow benchmarking tool.  

    The native benchmark binary for linux is available from this page:  
    [https://www.tensorflow.org/lite/performance/measurement](https://www.tensorflow.org/lite/performance/measurement)   

    Args:
        filepath (pathlib.Path): filepath to the extracted .txt file

    Returns:
        dict: dictionary with extracted model summary profiling information 
    """

    lines = clean_model_summary(filepath)
    model_profiling, operator_profiling = get_profiling_lines(lines)
    profiling_stats = get_model_profiling_stats(model_profiling)
    return profiling_stats

# Operator profiling

def get_operator_df(text, name=""):
    df = pd.DataFrame(text)
    df.rename(columns=df.iloc[0, :], inplace=True) 
    df.drop(df.index[0], inplace=True)
    df.columns = clean_column_names(df)
    # try:
    df["%"] = df["%"].apply(string_percent_to_float)
    # except:
    #     df["avg %"] = df["avg %"].apply(string_percent_to_float)
    # try:
    df["cdf%"] = df["cdf%"].apply(string_percent_to_float)
    # except:
    #     df["cdf %"] = df["cdf %"].apply(string_percent_to_float)
    df["first"] = df["first"].map(float)
    df["avg ms"] = df["avg ms"].map(float)
    df["mem KB"] = df["mem KB"].map(float)
    df["times called"] = df["times called"].map(int)#
    #df.Name = name

    return df

def get_node_df(text, name=""):
    df = pd.DataFrame(text)
    df.rename(columns=df.iloc[0, :], inplace=True) 
    df.drop(df.index[0], inplace=True)
    df.columns = clean_column_names(df)
    df["avg %"] = df["avg %"].apply(string_percent_to_float)
    df["cdf %"] = df["cdf %"].apply(string_percent_to_float)
    df["avg ms"] = df["avg ms"].map(float)
    df["mem KB"] = df["mem KB"].map(float)
    df["times called"] = df["times called"].map(int)
    df["count"] = df["count"].map(int)
    df.Name = name

    return df

def get_index_of_operator_profiling_lines(lines):
    operator_lines_dict = {}
    for i, line in enumerate(lines):
        #if line.str.contains('= Run Order ='):
        if "= Run Order =" in line:
            operator_lines_dict["run_order"] = i
        elif "= Top by Computation Time =" in line:
            operator_lines_dict["top_by_computation_time"] = i
        elif "= Summary by node type =" in line:
            operator_lines_dict["summary_by_node_type"] = i

    # print(operator_lines_dict)
    return operator_lines_dict

def get_summary_by_node_type_df(operator_profiling, operator_lines_dict, model_name):
    summary_by_node_type = operator_profiling[operator_lines_dict["summary_by_node_type"] +1:-3]
    summary_by_node_type = remove_tabs(summary_by_node_type)
    summary_by_node_type_df = get_node_df(summary_by_node_type, name=f"Summary by node type - {model_name}")
    return summary_by_node_type_df

def get_operator_top_by_comp_time_df(operator_profiling, operator_lines_dict, model_name):

    top_by_comp_time = operator_profiling[
        operator_lines_dict["top_by_computation_time"]+ 1 : operator_lines_dict["summary_by_node_type"]- 1]
    top_by_comp_time_clean = remove_tabs(top_by_comp_time)

    df = pd.DataFrame(top_by_comp_time_clean)

    df.rename(columns=df.iloc[0, :], inplace=True)
    df.drop(df.index[0], inplace=True)
    df.columns = clean_column_names(df)

    df["%"] = df["%"].apply(string_percent_to_float)
    df["cdf%"] = df["cdf%"].apply(string_percent_to_float)
    df["first"] = df["first"].map(float)
    df["avg ms"] = df["avg ms"].map(float)
    df["mem KB"] = df["mem KB"].map(float)
    df["times called"] = df["times called"].map(int)

    return df

def get_run_order_df(operator_lines_dict, operator_profiling, model_name):
    run_order = operator_profiling[operator_lines_dict["run_order"] +1: operator_lines_dict["top_by_computation_time"]]
    run_order = remove_tabs(run_order)
    df = get_operator_df(run_order, name= f"Run order - {model_name}")
    return df

def get_profiling_dataframes_cpu(filepath, model_name):
    """
    Parse "bechmarking.txt" file that has been generated from a .tflite model with TensorFlow benchmarking tool.  

    The native benchmark binary for linux is available from this page:  
    [https://www.tensorflow.org/lite/performance/measurement](https://www.tensorflow.org/lite/performance/measurement)   

    Args:
        filepath (pathlib.Path): filepath to the extracted .txt file

    Returns:
        list(pd.DataFrame, pd.DataFrame, pd.DataFrame): tuple containing the following dataframes:

            operator_run_order_df, operator_by_comp_time_df, summary_node_type_df
    """

    lines = clean_model_summary(filepath)
    model_profiling, operator_profiling = get_profiling_lines(lines)
    operator_lines_dict = get_index_of_operator_profiling_lines(operator_profiling)

    operator_run_order_df = get_run_order_df(operator_lines_dict, operator_profiling, model_name)
    operator_by_comp_time_df = get_operator_top_by_comp_time_df(operator_profiling, operator_lines_dict, model_name)
    summary_node_type_df = get_summary_by_node_type_df(operator_profiling, operator_lines_dict, model_name)

    return operator_run_order_df, operator_by_comp_time_df, summary_node_type_df