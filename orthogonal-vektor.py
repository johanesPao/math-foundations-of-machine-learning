import numpy as np

# ORTHOGONAL VECTORS
i = np.array([1, 0])
print(f'vektor i: {i}')

j = np.array([0, 1])
print(f'vektor j: {j}')

print('Vektor merupakan Orthogonal jika dot product dari satu set vektor adalah 0')
print(f'Dot product dari vektor i dan j adalah: {np.dot(i, j)}')
