References:
- https://github.com/SethHWeidman/DLFS_code/tree/master/lincoln/lincoln
- https://github.com/eriklindernoren/ML-From-Scratch
- https://github.com/ddbourgin/numpy-ml
- https://github.com/karpathy/micrograd
- https://github.com/geohot/tinygrad


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

