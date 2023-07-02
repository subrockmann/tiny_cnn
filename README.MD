# Tiny CNN - Developing Efficient CNN Architectures for Image Classification on Microcontroller Units 

This repository contains the notebooks for the CNN-Analyzer - a dashboard based tool for determining optimal CNN architecture parameters


| No. | File Name | Description |
| --- | --------- | ----------- |
| 01  | [01 Model Factory.ipynb](https://github.com/subrockmann/tiny_cnn/blob/master/01%20Model%20Factory.ipynb) | tf.model.summary<br>tf.keras.utils.plot_model<br>Visualization of model graph<br>Untrained TensorFlow model |
| 02  | [02 TensorFlow Layer Analyzer.ipynb](https://github.com/subrockmann/tiny_cnn/blob/master/02%20TensorFlow%20Layer%20Analyzer.ipynb) | tf.model.layers layerwise model info |
| 03  | [03 Model Analyzer.ipynb](https://github.com/subrockmann/tiny_cnn/blob/master/03%20Model%20Analyzer.ipynb) | Parsing <modelname>.txt<br>MACs<br>FLOPs<br>Total parameters<br>Trainable parameters<br>Non-trainable parameters<br>Create dataframe from <modelname>_layers.csv<br>Untrained TensorFlow model conversion to TFLite<br>TFLite model<br>Untrained TensorFlow model conversion to TFLite with int8-quantization<br>Int8 quantized TFLite model<br>TFLite model size in KB<br>TFLite int8 model size in KB<br>tflite_tools.py (E. Liberis) layerwise RAM in bytes<br>Max value from layerwise RAM in bytes<br>General model architecture parameters<br>Model name<br>Model variation<br>Alpha factor<br>Image resolution<br>Number of classes<br>Number of input channels (RGB or grayscale)<br>MLTK Profiler (hardware simulation) layerwise metrics (ops, MACs, CPU cycles, energy, input shape and output shape)<br>Model size in bytes<br>RAM in bytes<br>FLOPs<br>MACs<br>Number of unsupported layers<br>Int8 energy |
| 04  | [04 Deploy to MCU.ipynb](https://github.com/subrockmann/tiny_cnn/blob/master/04%20Deploy%20to%20MCU.ipynb) | Convert TFLite model to C code |
| 05  | [05 LINUX - Benchmarking TFLite Models.ipynb](https://github.com/subrockmann/tiny_cnn/blob/master/05%20LINUX%20-%20Benchmarking%20TFLite%20Models.ipynb) | TensorFlow native benchmarking tool<br>Layerwise benchmarking report of int8 model running on desktop |
| 06  | [06 LINUX - Parsing CPU Benchmark Data.ipynb](https://github.com/subrockmann/tiny_cnn/blob/master/06%20LINUX%20-%20Parsing%20CPU%20Benchmark%20Data.ipynb) | Parsing <modelname>_benchmark.txt |
| 07  | [07 Model Training.ipynb](https://github.com/subrockmann/tiny_cnn/blob/master/07%20Model%20Training.ipynb) | Trained TensorFlow model<br>TFLite model<br>Int8 quantized TFLite model |
| 08  | [08 Update Model DB.ipynb](https://github.com/subrockmann/tiny_cnn/blob/master/08%20Update%20Model%20DB.ipynb) | Write test accuracy from model training project to the model database |
| 09  | [09 Benchmarking on STM32-Cube.AI.ipynb]() | Requires profiling int8 quantized with STM32-Cube.AI on external MCU<br>Parsing <model_name>_stm32_benchmark/network_validate_report_INT8.txt<br>CPU cycles per MAC<br>Inference time in milliseconds<br>Inference duration and layer type per layer |
| 10  | [10 Model Comparison & Visualizations.ipynb](https://github.com/subrockmann/tiny_cnn/blob/master/10%20Model%20Comparison%20%26%20Visualizations.ipynb) | Create visualizations |

<br>  

# Install the Conda Environment

```conda env create -f environment.yml
conda env create -f conda-environment.yhttps://github.com/subrockmann/tiny_cnn/blob/master/08%20Update%20Model%20DB.ipynb
```

## Activate the Environmnet

```
conda activate tiny_cnn
```


## Update the Environment
```
conda activate tiny_cnn
conda env update --file environment.yml # --prune
```

## Uninstalling the Environment  
If there are too many error about inconsistencies, it might be easier to uninstall the environment and re-install it from scratch.  
```
conda deactivate
conda env remove --name tiny_cnn
```

## If Nothing Else Helps
```
conda clean --all
```

## You Might Want to Run:
```
conda update --all
```

## Libcupti Issue:
```
conda install -c nvidia cuda-cupti
```
<>
# Setting up the Arduino IDE

Unfortunately the Arduino_TensorFlowLite library is not include in the Arduino Library Manager. Instead the library has to be installed from a ZIP File 
[https://www.ardu-badge.com/Arduino_TensorFlowLite/zip](https://www.ardu-badge.com/Arduino_TensorFlowLite/zip)  
