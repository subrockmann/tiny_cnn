import sys, os, csv
#import pathlib
from pathlib import Path
 
# setting path
sys.path.append('..')
 

from workbench.config.config import initialize
models_dir = initialize()

# global models_summary_path 
# models_summary_path = models_dir.joinpath(model_name, f"{model_name}.txt")

# global models_image_path
# models_image_path = models_dir.joinpath(model_name, f"{model_name}.png")

# global models_layer_df_path
# models_layer_df_path = models_dir.joinpath(model_name, f"{model_name}_layers.pkl")

# global models_tf_path
# models_tf_path = models_dir.joinpath(model_name, f"{model_name}.h5")

# global models_tflite_path
# models_tflite_path = models_dir.joinpath(model_name, f"{model_name}.tflite")

# global models_tflite_opt_path
# models_tflite_opt_path = models_dir.joinpath(model_name, f"{model_name}_INT8.tflite")

# #models_summary_path 

def create_filepaths(model_name):
    global models_dir
    print(models_dir)
    models_path = models_dir.joinpath(model_name)
    if not models_path.exists():
        print(f"{models_path} does not exist.")
        models_path.mkdir()
        print(f"Created path: {models_path}.")

    global models_summary_path 
    models_summary_path = models_dir.joinpath(model_name, f"{model_name}.txt")

    global models_image_path
    models_image_path = models_dir.joinpath(model_name, f"{model_name}.png")

    global models_layer_df_path
    models_layer_df_path = models_dir.joinpath(model_name, f"{model_name}_layers.csv")
    
    global models_tf_path
    models_tf_path = models_dir.joinpath(model_name, f"{model_name}.h5")

    global models_tflite_path
    models_tflite_path = models_dir.joinpath(model_name, f"{model_name}.tflite")
    
    global models_tflite_opt_path
    models_tflite_opt_path = models_dir.joinpath(model_name, f"{model_name}_INT8.tflite")
    
    return (models_path, models_summary_path, models_image_path, models_layer_df_path, models_tf_path, models_tflite_path, models_tflite_opt_path)

# create_filepaths()


def get_file_size(filepath):
    # get the file size of a file saved on operating system
    file_stats = os.stat(filepath)
    file_size_kb = file_stats.st_size / 1024
    print(f'File size in bytes is {file_stats.st_size}')
    print(f'File size in kilobytes is {file_size_kb}')
    return file_size_kb

def parse_model_name(model_name):
    """"""
    """Split the encoded model name into the parameters

    Args:
        model_name (string): Encoded model name

    Returns:
        string: Base model name
    """
    global base_model_name
    global alpha
    global resolution
    global channels
    global classes
    global variation
    base_model_name, alpha, resolution, channels, classes, variation = model_name.split("_")
    return base_model_name, alpha, resolution, channels, classes, variation


def create_model_name(base_model_name, alpha, input_shape, classes, variation_code):
    """This function returns a concatenated modelname that contains the model parameters as input.

    Args:
        base_model_name (string): Name of the model architecture e.g. MobileNetV1
        alpha (float): Value between 0 and 1 for the width factor of the model architecture
        input_shape (Tuple(int, int, int)): Input resolution of the model (img_width, img_height, channels)
        classes (int): Number of output classes of the classifier
        variation_code (string): Encoding for additionally needed remarks. (Default: "000")
            (Important: Do not use "_" in this part as it will break the reverse function!)

    Returns:
        string: Encoded model name
    """
    model_name = f"{base_model_name}_{alpha}_{input_shape[0]}_c{input_shape[2]}_o{classes}_{variation_code}"
    return model_name

def append_dict_to_csv(csv_path, data_dict):
    """Appends dictionary to a csv file. 
    If the csv file does not exist, the file will be created and the headers are written to the file. 

    Args:
        csv_path (string: Filepath to the location of the csv file.
        data_dict (dict): Dictionary that is appended to the file
    """
    if not csv_path.exists():
        print(f"{csv_path} does not exist.")
        
        with open(csv_path, "a",  newline='') as file:
            writer = csv.DictWriter(file, fieldnames =data_dict.keys())
            writer.writeheader()
            writer.writerow(data_dict) 
            print(f"Created file: {csv_path} inlcuding headers.")
    else:
        with open(csv_path, "a",  newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data_dict.keys())
            writer.writerow(data_dict) 

    return


def parse_mltk_model_summary(filepath): 
    # Parse the MLTK model summary to grab important metrics   
    with open(filepath, "r") as f:
        lines = f.readlines() # list containing lines of file
        #columns = [] # To store column names

        i = 1
        for line in lines:
            line = line.strip() # remove leading/trailing white spaces
            if line.startswith("Total params:"):
                total_params = line.split()[-1]
                total_params = int(total_params.replace(",", ""))
            elif line.startswith("Trainable params:"):
                trainable_params = line.split()[-1]
                trainable_params =  int(trainable_params.replace(",", ""))
            elif line.startswith("Non-trainable params:"):
                non_trainable_params = line.split()[-1]
                non_trainable_params = int(non_trainable_params.replace(",", ""))
            elif line.startswith("Total MACs:"):
                MACs = line.split()[-2] + " " + line.split()[-1]
                #MACs = (float(MACs))
            elif line.startswith("Total OPs:"):
                FLOPs = line.split()[-2] + " " + line.split()[-1]
                #FLOPs = (float(FLOPs))
            else:
                pass
    
    return (total_params, trainable_params, non_trainable_params, MACs, FLOPs)
