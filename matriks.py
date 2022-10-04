import torch
import tensorflow as tf
import numpy as np

# Use array() with nested brackets:
X = np.array([[25, 2], [5, 26], [3, 7]])
print(X)
print(f'Shape dari matriks X: {X.shape}')
print(f'Ukuran dari matriks X: {X.size}')

# Select left column of matrix X (zero-based):
print('Kolom kiri dari matriks X:')
print([X[:, 0]])

# Select middle row of matrix X:
print('Baris kedua dari matriks X:')
print(X[1, :])

# Another slicing-by-index example:
print(
    'Slice matriks untuk mengembalikan dua baris dan dua kolom pertama X[0:2, 0:2]:')
print(X[0:2, 0:2])

# PyTorch
X_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])
print(X_pt)
print(f'Shape dari matriks X (PyTorch): {X_pt.shape}')

print(
    'Slice matriks untuk mengembalikan baris kedua X_pt[1, :]:')
print(X_pt[1, :])

# TensorFlow
X_tf = tf.Variable([[25, 2], [5, 26], [3, 7]])
print(X_tf)
print(f'Rank dari tensor matriks: {tf.rank(X_tf)}')
print(f'Shape dari matriks X (TensorFlow): {tf.shape(X_tf)}')

print(
    'Slice matriks untuk mengembalikan baris kedua X_tf[1, :]:')
print(X_tf[1, :])

images_pt = torch.zeros([32, 28, 28, 3])
images_tf = tf.zeros([32, 28, 28, 3])

print(images_pt)
print(images_tf)
