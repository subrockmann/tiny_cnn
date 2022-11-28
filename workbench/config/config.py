from pathlib import Path

global models_dir
# global models_summary_path 
# global models_image_path
# global models_layer_df_path
# global models_tf_path
# global models_tflite_path
# global models_tflite_opt_path

def initialize():
    global models_dir
    models_dir = Path.cwd().joinpath("models")
    if not models_dir.exists():
        print(f"{models_dir} does not exist.")
        models_dir.mkdir()
        print(f"Created path: {models_dir}.")
    return models_dir

#initialize()