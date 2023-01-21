arduino-cli compile --fqbn arduino:mbed_nano:nano33ble model_test

bash: rduino-cli: command not found
(tiny_cnn_5) 
Susanne@HP-Omen MINGW64 /i/tinyml/tiny_cnn/deployment (master)
$ arduino-cli compile --fqbn arduino:mbed_nano:nano33ble model_test
I:\tinyml\tiny_cnn\deployment\model_test\model_test.ino:42:1: error: 'error_reporter' does not name a type
 error_reporter = &micro_error_reporter;
 ^~~~~~~~~~~~~~
I:\tinyml\tiny_cnn\deployment\model_test\model_test.ino:44:53: error: no matching function for call to 'tflite::MicroProfiler::MicroProfiler(tflite::MicroErrorReporter*)'
 tflite::MicroProfiler profiler(&micro_error_reporter)
                                                     ^
In file included from I:\tinyml\tiny_cnn\deployment\model_test\model_test.ino:27:0:
I:\tinyml\tiny_vision_mcu\libraries\tflite-micro-arduino-examples\src/tensorflow/lite/micro/micro_profiler.h:30:3: note: candidate: tflite::MicroProfiler::MicroProfiler()
   MicroProfiler() = default;
   ^~~~~~~~~~~~~
I:\tinyml\tiny_vision_mcu\libraries\tflite-micro-arduino-examples\src/tensorflow/lite/micro/micro_profiler.h:30:3: note:   candidate expects 0 arguments, 1 provided
In file included from I:\tinyml\tiny_cnn\deployment\model_test\model_test.ino:27:0:
I:\tinyml\tiny_vision_mcu\libraries\tflite-micro-arduino-examples\src/tensorflow/lite/micro/micro_profiler.h:28:7: note: candidate: constexpr tflite::MicroProfiler::MicroProfiler(const tflite::MicroProfiler&)
 class MicroProfiler : public MicroProfilerInterface {
       ^~~~~~~~~~~~~
I:\tinyml\tiny_vision_mcu\libraries\tflite-micro-arduino-examples\src/tensorflow/lite/micro/micro_profiler.h:28:7: note:   no known conversion for argument 1 from 'tflite::MicroErrorReporter*' to 'const tflite::MicroProfiler&'
I:\tinyml\tiny_cnn\deployment\model_test\model_test.ino:46:14: error: expected ',' or ';' before 'constexpr'
              constexpr int kTensorArenaSize = 2000;
              ^~~~~~~~~
I:\tinyml\tiny_cnn\deployment\model_test\model_test.ino:47:22: error: 'kTensorArenaSize' was not declared in this scope
 uint8_t tensor_arena[kTensorArenaSize];
                      ^~~~~~~~~~~~~~~~
I:\tinyml\tiny_cnn\deployment\model_test\model_test.ino: In function 'void setup()':
I:\tinyml\tiny_cnn\deployment\model_test\model_test.ino:80:24: error: 'tensor_arena' was not declared in this scope
       model, resolver, tensor_arena, kTensorArenaSize &micro_error_reporter, &profiler); //, error_reporter);
                        ^~~~~~~~~~~~
I:\tinyml\tiny_cnn\deployment\model_test\model_test.ino:80:38: error: 'kTensorArenaSize' was not declared in this scope
       model, resolver, tensor_arena, kTensorArenaSize &micro_error_reporter, &profiler); //, error_reporter);
                                      ^~~~~~~~~~~~~~~~


Used library           Version     Path
Arduino_BuiltIn        1.0.0       I:\tinyml\tiny_vision_mcu\libraries\Arduino_BuiltIn
Arduino_TensorFlowLite 2.4.0-ALPHA I:\tinyml\tiny_vision_mcu\libraries\tflite-micro-arduino-examples
Wire                               C:\Users\Susanne\AppData\Local\Arduino15\packages\arduino\hardware\mbed_nano\3.5.4\libraries\Wire

Used platform     Version Path
arduino:mbed_nano 3.5.4   C:\Users\Susanne\AppData\Local\Arduino15\packages\arduino\hardware\mbed_nano\3.5.4

Error during build: exit status 1
(tiny_cnn_5) 
Susanne@HP-Omen MINGW64 /i/tinyml/tiny_cnn/deployment (master)
$