# Skalar (Rank 0 tensor) di base Python
import tensorflow as tf
import torch
x = 25

# 25
print(x)

# tipe dari x
# int
print(type(x))

y = 3

py_sum = x + y

# 28
print(py_sum)

# tipe dari py_sum
# int
print(type(py_sum))

x_float = 25.0
float_sum = x_float + y

# tipe dari float_sum
# float
# float + int = float
print(type(float_sum))

# Skalar dalam PyTorch
# - PyTorch didesain dengan sifat yang pythonic dengan sifat dan perilaku yang menyerupai array pada NumPy
# - Keunggulan dari tensor PyTorch dibandingkan dengan array pada NumPy adalah bahwa tensor PyTorch dapat dioperasikan menggunakan GPU
# - Dokumentasi untuk tensor PyTorch, termasuk dengan tipe datanya bisa dilihat di https://pytorch.org/docs/stable/tensors.html
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

x_pt = torch.tensor(25)

print(type(x_pt))

# Skalar dalam TensorFlow

x_tf = tf.Variable(25, dtype=tf.int16)  # dtype is optional
print(x_tf)

print(x_tf.shape)

y_tf = tf.Variable(3, dtype=tf.int16)
print(x_tf + y_tf)

tf_sum = tf.add(x_tf, y_tf)
print(tf_sum)

tf_sum.numpy()

# note that Numpy operation automatically convert tensors to NumPy arrays, amd vice versa
print(type(tf_sum.numpy()))

tf_float = tf.Variable(25, dtype=tf.float16)
print(tf_float)
