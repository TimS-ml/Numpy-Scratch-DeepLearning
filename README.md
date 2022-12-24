```
                                                  d8b        
                                     d8P          ?88        
                                  d888888P         88b       
 .d888b, d8888b  88bd88b d888b8b    ?88'   d8888b  888888b   
 ?8b,   d8P' `P  88P'  `d8P' ?88    88P   d8P' `P  88P `?8b  
   `?8b 88b     d88     88b  ,88b   88b   88b     d88   88P  
`?888P' `?888P'd88'     `?88P'`88b  `?8b  `?888P'd88'   88b  
```


# About This Repo
So I like https://github.com/ddbourgin/numpy-ml and https://github.com/geohot/tinygrad but they are not simple enough

The Ultimate Goal:
- Keep the dependency simple (mainly numpy)
- Eazy migration from PyTorch, while keep the code structure simple
- Support datasets from sklearn and torch
- Implement models with the complexity of YOLO and Transformer
- Numpy GPU acceleration (so decient speed)
- C++ version


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
