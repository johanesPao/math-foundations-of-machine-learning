import numpy as np
import torch
import tensorflow as tf

X = np.array([[25, 2], [5, 26], [3, 7]])
X_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])
X_tf = tf.Variable([[25, 2], [5, 26], [3, 7]])

print('X:')
print(X)

# TRANSPOSISI TENSOR

print('Transposisi X:')
print(X.T)

print('Transposisi X_tf (TensorFlow):')
print(tf.transpose(X_tf))

print('Transposisi X_pt (PyTorch):')
print(X_pt.T)

# ARITMATIKA DASAR
print('X*2:')
print(X*2)
print('X+2:')
print(X+2)
print('X*2+2:')
print(X*2+2)
print('X_pt*2+2: (sama dengan menggunakan torch.mul() dan torch.add())')
print(X_pt*2+2)
print('torch.add(torch.mul(X_pt, 2), 2):')
print(torch.add(torch.mul(X_pt, 2), 2))
print('X_tf*2+2: (sama dengan menggunakan tf.multiply() dan tf.add()')
print(X_tf*2+2)
print('tf.add(tf.multiply(X_tf, 2), 2):')
print(tf.add(tf.multiply(X_tf, 2), 2))

# HADAMARD PRODUCT
A = X + 2
A_pt = X_pt + 2
A_tf = X_tf + 2
print('A:')
print(A)
print('A + X:')
print(A + X)
print('A * X:')
print(A * X)
print('A_pt + X_pt:')
print(A_pt + X_pt)
print('A_pt * X_pt:')
print(A_pt * X_pt)
print('A_tf + X_tf:')
print(A_tf + X_tf)
print('A_tf * X_tf:')
print(A_tf * X_tf)

# REDUCTION
print('X:')
print(X)
print('X.sum():')
print(X.sum())
print('torch.sum(X_pt):')
print(torch.sum(X_pt))
print('tf.reduce_sum(X_tf):')
print(tf.reduce_sum(X_tf))
print('Dapat juga dilakukan pada spesifik axis, contohnya:')
print('X.sum(axis=0): # Penjumlahan secara vertikal terhadap semua baris dalam masing - masing kolom')
print(X.sum(axis=0))
print('X.sum(axis=1): # Penjumlahan secara horizontal terhadap semua kolom dalam masing - masing baris')
print(X.sum(axis=1))
print('torch.sum(X_pt, 0):')
print(torch.sum(X_pt, 0))
print('tf.reduce_sum(X_tf, 1):')
print(tf.reduce_sum(X_tf, 1))

# DOT PRODUCT
x = np.array([25, 2, 5])
y = np.array([0, 1, 2])
x_pt = torch.tensor([25, 2, 5])
y_pt = torch.tensor([0, 1, 2])
x_tf = tf.Variable([25, 2, 5])
y_tf = tf.Variable([0, 1, 2])
print('x:')
print(x)
print('y:')
print(y)
print('25*0 + 2*1 + 5*2:')
print(25*0 + 2*1 + 5*2)
print('np.dot(x, y):')
print(np.dot(x, y))
print('y_pt:')
print(y_pt)
print('np.dot(x_pt, y_pt):')
print(np.dot(x_pt, y_pt))
print('Atau kita dapat menggunakan torch.dot(), namun tensor harus dalam bentuk float')
print('torch.dot(torch.tensor([25, 2, 5.]), torch.tensor([0, 1, 2.]))):')
print(torch.dot(torch.tensor([25, 2, 5.]), torch.tensor([0, 1, 2.])))
print('tf.reduce_sum(tf.multiply(x_tf, y_tf)):')
print(tf.reduce_sum(tf.multiply(x_tf, y_tf)))
