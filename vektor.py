# Vector (Rank 1 Tensors) in NumPy
import tensorflow as tf
import torch
import numpy as np

x = np.array([25, 2, 1])  # type argument is optional, e.g.: dtype=np.float16
print(x)
print(len(x))
print(x.shape)
print(type(x))

# accessing vector element using index
print(x[0])

# type of element of a vector is a scalar
print(type(x[0]))

# vector transposition
x_t = x.T
print(x_t)
print(x_t.shape)
print('regular 1-D array define as x above doesn\'t have any effect when being transposed, due to it\'s shape (3,) which only has 1 dimension and no other dimension to transpose to...')

y = np.array([[25, 2, 5]])
print(y)
print(y.shape)

y_t = y.T
print(y_t)
print(y_t.shape)
print('this column vector can be converted back to row vector using the same operation of transposition')
print(y_t.T.shape)

z = np.zeros(3)
print(z)
print('this are zeros vector which has no effect on vector addition, but might be useful for other kind of vector operation')


x_pt = torch.tensor([25, 2, 5])
print(x_pt)

x_tf = tf.Variable([25, 2, 5])
print(x_tf)
