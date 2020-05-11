# This checks if pytorch is using GPU
import torch
assert torch.cuda.is_available(), "Cuda not available"

# This checks if mxnet is using GPU
import mxnet as mx
a = mx.nd.ones((2, 3), mx.gpu())
b = a * 2 + 1
b.asnumpy()
array([[ 3.,  3.,  3.],[ 3.,  3.,  3.]], dtype=float32)
