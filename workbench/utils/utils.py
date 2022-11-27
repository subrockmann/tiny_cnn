import sys
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
    models_layer_df_path = models_dir.joinpath(model_name, f"{model_name}_layers.pkl")
    
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