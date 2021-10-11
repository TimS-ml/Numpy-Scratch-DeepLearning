```
                                                  d8b        
                                     d8P          ?88        
                                  d888888P         88b       
 .d888b, d8888b  88bd88b d888b8b    ?88'   d8888b  888888b   
 ?8b,   d8P' `P  88P'  `d8P' ?88    88P   d8P' `P  88P `?8b  
   `?8b 88b     d88     88b  ,88b   88b   88b     d88   88P  
`?888P' `?888P'd88'     `?88P'`88b  `?8b  `?888P'd88'   88b  
```


# About this repo
I want to run machine learning and deep leanring on my arm devices for fun (iPad, Android Phone, Raspberry pi)

The ultimate goal is to achieve something like [numpy-ml](https://github.com/ddbourgin/numpy-ml/tree/master/numpy_ml/neural_nets) and [tinygrad](https://github.com/geohot/tinygrad)

References:
- https://github.com/SethHWeidman/DLFS_code/tree/master/lincoln/lincoln
- https://github.com/eriklindernoren/ML-From-Scratch
- https://github.com/ddbourgin/numpy-ml
- https://github.com/karpathy/micrograd
- https://github.com/geohot/tinygrad


Check my other repos: 
- https://github.com/TimS-ml/Scratch-ML
- https://github.com/TimS-ml/Scratch-DL
- https://github.com/TimS-ml/My-ML
- https://github.com/TimS-ml/My-Algo



# Quick Start
```python
from scratchDL import base as dl
from scratchDL.base import layers as lyr
from scratchDL.base import optm
from scratchDL.base import loss
from scratchDL.base import activation as act
from scratchDL.base import NeuralNetwork


clf = dl.NeuralNetwork(
            optimizer=optm.Adam(),  # default lr is 0.001
            loss=loss.CrossEntropy,
            validation_data=(X_test, y_test))

clf.add(lyr.Conv2D(
              n_filters=16,
              filter_shape=(3, 3),
              stride=1,
              input_shape=(1, 8, 8),
              padding='same'))
clf.add(lyr.Activation(act.ReLU))
clf.add(lyr.Dropout(0.25))
clf.add(lyr.BatchNorm())
clf.add(lyr.Conv2D(
              n_filters=32, 
              filter_shape=(3, 3), 
              stride=1,
              padding='same'))
clf.add(lyr.Activation(act.ReLU))
clf.add(lyr.Dropout(0.25))
clf.add(lyr.BatchNorm())
clf.add(lyr.Flatten())
clf.add(lyr.Dense(256))
clf.add(lyr.Activation(act.ReLU))
clf.add(lyr.Dropout(0.4))
clf.add(lyr.BatchNorm())
clf.add(lyr.Dense(10))
clf.add(lyr.Activation(act.Softmax))
```


# Data Inputs
You can find dataset here:
- `sklearn.datasets`
- `torch`, `torchtext`, `torchvision`
- `gym`
- `pyglet`


# TODO
- [ ] Features
  - [ ] Weight Init Func
- [ ] Re-Design
  - [ ] Batch / Channel / 1-dim shape
  - [ ] Param / NonParam Layers
  - [ ] PyOpenCL or CuPy for Nvidia and AMD GPU
    - [ ] `scratchDL.Tensor` and `Tensor.gpu`
    - check: https://github.com/geohot/tinygrad/blob/master/tinygrad/ops_gpu.py
  - [ ] CPU Parallel Programming with numpy and scipy
- [ ] Add more models
  - [ ] Transformer
  - [ ] VGG
  - [ ] Inception
  - [ ] Efficientnet
- [x] Reduce package usage
  - [x] Update `setup.py` with `setup_require` and `extras_require`

## Pending
- [ ] Roughly compatible with PyTorch and sk-learn
  - [ ] Function name, operations, forward, backward etc.
    - [ ] Re-write implementations of `scratchDL.base`
