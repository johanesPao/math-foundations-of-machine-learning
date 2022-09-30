# Skalar (Rank 0 tensor) di base Python
x = 25

# 25
print(x)

# tipe dari x
# int
type(x)

y = 3

py_sum = x + y

# 28
print(py_sum)

# tipe dari py_sum
# int
type(py_sum)

x_float = 25.0
float_sum = x_float + y

# tipe dari float_sum
# float
# float + int = float
type(float_sum)

# Skalar dalam PyTorch
# - PyTorch didesain dengan sifat yang pythonic dengan sifat dan perilaku yang menyerupai array pada NumPy
# - Keunggulan dari tensor PyTorch dibandingkan dengan array pada NumPy adalah bahwa tensor PyTorch dapat dioperasikan menggunakan GPU
# - Dokumentasi untuk tensor PyTorch, termasuk dengan tipe datanya bisa dilihat di https://pytorch.org/docs/stable/tensors.html

import torch
x_pt = torch.tensor(25)

# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113