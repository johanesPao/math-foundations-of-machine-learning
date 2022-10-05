import numpy as np
import torch
import tensorflow as tf

X = np.array([[1, 2], [3, 4]])
X_pt = torch.tensor([[1, 2], [3, 4.]])  # torch.norm() requires float type
X_tf = tf.Variable([[1, 2], [3, 4.]])  # tf.norm() also requires float type

# FROBENIUS FORM
print('----FROBENIUS FORM')
print('\nX:')
print(X)
print('\n(1**2 + 2**2 + 3**2 + 4**2)**(1/2):')
print((1**2 + 2**2 + 3**2 + 4**2)**(1/2))
print('\nnp.linalg.norm(X):')
print(np.linalg.norm(X))
print('\nX_pt:')
print(X_pt)
print('\ntorch.norm(X_pt):')
print(torch.norm(X_pt))
print('\nX_tf:')
print(X_tf)
print('\ntf.norm(X_tf):')
print(tf.norm(X_tf))

# MATRIX MULTIPLICATION
A = np.array([[3, 4], [5, 6], [7, 8]])
b = np.array([1, 2])
A_pt = torch.tensor([[3, 4], [5, 6], [7, 8]])
b_pt = torch.tensor([1, 2])
A_tf = tf.Variable([[3, 4], [5, 6], [7, 8]])
b_tf = tf.Variable([1, 2])

print('\n---MATRIX MULTIPLICATION')
print('\nA:')
print(A)
print('\nb:')
print(b)
print('\nnp.dot(A, b):')
print(np.dot(A, b))
print('\nA_pt:')
print(A_pt)
print('\nb_pt:')
print(b_pt)
print('\ntorch.matmul(A, b_pt):')
print(torch.matmul(A_pt, b_pt))
print('\nA_tf:')
print(A_tf)
print('\nb_tf:')
print(b_tf)
print('\ntf.linalg.matvec(A, b_tf):')
print(tf.linalg.matvec(A_tf, b_tf))

B = np.array([[1, 9], [2, 0]])
B_pt = torch.from_numpy(B.astype(np.int64))
B_tf = tf.convert_to_tensor(B, dtype=tf.int32)

print('\nB:')
print(B)
print('\nnp.dot(A, B):')
print(np.dot(A, B))
print('\nB_pt')
print(B_pt)
print('\ntorch.matmul(A_pt, B_pt):')
print(torch.matmul(A_pt, B_pt))
print('\nB_tf:')
print(B_tf)
print('\ntf.matmul(A_tf, B_tf):')
print(tf.matmul(A_tf, B_tf))
