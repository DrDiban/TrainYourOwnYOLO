---
name: Bug Report or Feature Request
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---
Before filing a report consider the following two questions:

### Have you followed the instructions exactly (word by word)?

### Have you checked the [troubleshooting](https://github.com/AntonMu/TrainYourOwnYOLO#troubleshooting) section?

Once you are familiar with the code, you're welcome to modify it. Please only continue to file a bug report if you encounter an issue with the provided code and after having followed the instructions.

If you have followed the instructions exactly, couldn't solve your problem with the provided troubleshooting tips and would still like to file a bug or make a feature requests please follow the steps below.

1. It must be a bug, a feature request, or a significant problem with the documentation (for small docs fixes please send a PR instead).
2. The form below must be filled out.

------------------------

### System information
- **What is the top-level directory of the model you are using**:C:\
- **Have I written custom code (as opposed to using a stock example script provided in the repo)**:No
- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**:Windows 10 Intel core I5-6200 2.30GHz Ram 4GB
- **TensorFlow version (use command below)**:v1.15.0-rc3-22-g590d6eef7e 1.15.0
- **CUDA/cuDNN version**:Not using GPU
- **GPU model and memory**:Not using GPU
- **Exact command to reproduce**:python Train_YOLO.py 

You can obtain the TensorFlow version with

`python -c "import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)"`

### Describe the problem
There were errors/warnings produced when command python Train_YOLO.py was excecuted. The model train for a while on epoch 1/51 and then was halted.

### Source code / logs
Create YOLOv3 model with 9 anchors and 1 classes.
C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\saving.py:1140: UserWarning: Skipping loading of weights for layer conv2d_59 due to mismatch in shape ((1, 1, 1024, 18) vs (255, 1024, 1, 1)).
  weight_values[i].shape))
C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\saving.py:1140: UserWarning: Skipping loading of weights for layer conv2d_59 due to mismatch in shape ((18,) vs (255,)).
  weight_values[i].shape))
C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\saving.py:1140: UserWarning: Skipping loading of weights for layer conv2d_67 due to mismatch in shape ((1, 1, 512, 18) vs (255, 512, 1, 1)).
  weight_values[i].shape))
C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\saving.py:1140: UserWarning: Skipping loading of weights for layer conv2d_67 due to mismatch in shape ((18,) vs (255,)).
  weight_values[i].shape))
C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\saving.py:1140: UserWarning: Skipping loading of weights for layer conv2d_75 due to mismatch in shape ((1, 1, 256, 18) vs (255, 256, 1, 1)).
  weight_values[i].shape))
C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\saving.py:1140: UserWarning: Skipping loading of weights for layer conv2d_75 due to mismatch in shape ((18,) vs (255,)).
  weight_values[i].shape))
Load weights C:\TrainYourOwnYOLO\2_Training\src\keras_yolo3\yolo.h5.
Freeze the first 249 layers of total 252 layers.


Epoch 1/51
C:\TrainYourOwnYOLO\env\lib\site-packages\PIL\Image.py:989: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  "Palette images with Transparency expressed in bytes should be "
2020-01-27 16:08:13.307869: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] shape_optimizer failed: Invalid argument: Subshape must have computed start >= end since stride is negative, but is 0 and 2 (computed from start 0 and end 9223372036854775807 over shape with rank 2 and stride-1)
2020-01-27 16:08:13.507147: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: Subshape must have computed start >= end since stride is negative, but is 0 and 2 (computed from start 0 and end 9223372036854775807 over shape with rank 2 and stride-1)
2020-01-27 16:08:14.696568: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] shape_optimizer failed: Invalid argument: Subshape must have computed start >= end since stride is negative, but is 0 and 2 (computed from start 0 and end 9223372036854775807 over shape with rank 2 and stride-1)
2020-01-27 16:08:14.941158: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: Subshape must have computed start >= end since stride is negative, but is 0 and 2 (computed from start 0 and end 9223372036854775807 over shape with rank 2 and stride-1)
2020-01-27 16:08:16.656976: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 708837376 exceeds 10% of system memory.
2020-01-27 16:08:21.208633: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 708837376 exceeds 10% of system memory.
2020-01-27 16:08:43.647250: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 712249344 exceeds 10% of system memory.
2020-01-27 16:08:46.577161: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 354418688 exceeds 10% of system memory.
2020-01-27 16:08:53.241032: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 177209344 exceeds 10% of system memory.
1/2 [==============>...............] - ETA: 3:01 - loss: 12746.64552020-01-27 16:13:34.665739: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: Subshape must have computed start >= end since stride is negative, but is 0 and 2 (computed from start 0 and end 9223372036854775807 over shape with rank 2 and stride-1)
2020-01-27 16:13:35.499608: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: Subshape must have computed start >= end since stride is negative, but is 0 and 2 (computed from start 0 and end 9223372036854775807 over shape with rank 2 and stride-1)
2020-01-27 16:14:08.879664: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at pad_op.cc:137 : Resource exhausted: OOM when allocating tensor with shape[32,417,417,32] and type float on /job:localhost/replica:0/task:0/device:CPU:0 by allocator cpu
Traceback (most recent call last):
  File "Train_YOLO.py", line 217, in <module>
    callbacks=[logging, checkpoint],
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\keras\legacy\interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\training.py", line 1418, in fit_generator
    initial_epoch=initial_epoch)
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\training_generator.py", line 234, in fit_generator
    workers=0)
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\keras\legacy\interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\training.py", line 1472, in evaluate_generator
    verbose=verbose)
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\training_generator.py", line 346, in evaluate_generator
    outs = model.test_on_batch(x, y, sample_weight=sample_weight)
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\keras\engine\training.py", line 1256, in test_on_batch
    outputs = self.test_function(ins)
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\keras\backend\tensorflow_backend.py", line 2715, in __call__
    return self._call(inputs)
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\keras\backend\tensorflow_backend.py", line 2675, in _call
    fetched = self._callable_fn(*array_vals)
  File "C:\TrainYourOwnYOLO\env\lib\site-packages\tensorflow_core\python\client\session.py", line 1472, in __call__
    run_metadata_ptr)
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[32,417,417,32] and type float on /job:localhost/replica:0/task:0/device:CPU:0 by allocator cpu
         [[{{node zero_padding2d_1/Pad}}]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
