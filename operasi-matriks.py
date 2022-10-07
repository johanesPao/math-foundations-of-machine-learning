import numpy as np
import matplotlib.pyplot as plt

# AFFINE TRANSFORMATION VIA MATRIX APPLICATION
print('\n---AFFINE TRANSFORMATION VIA MATRIX APPLICATION')
v = np.array([3, 1])
print('\nLet\'s say we have a vector v')
print('v:')
print(v)


def plot_vectors(vectors, colors):
    """
    Plot one or more vectors in a 2D plane, specifying a color for each. 

    Arguments
    ---------
    vectors: list of lists or of arrays
        Coordinates of the vectors to plot. For example, [[1, 3], [2, 2]] 
        contains two vectors to plot, [1, 3] and [2, 2].
    colors: list
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.

    Example
    -------
    plot_vectors([[1, 3], [2, 2]], ['red', 'blue'])
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    """
    plt.figure()
    plt.axvline(x=0, color='lightgray')
    plt.axhline(y=0, color='lightgray')

    for i in range(len(vectors)):
        x = np.concatenate([[0, 0], vectors[i]])
        plt.quiver([x[0]], [x[1]], [x[2]], [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=colors[i],)


print('\nMenggunakan fungsi plot_vectors() yang dibuat oleh Hadrien Jean')
print('plot_vectors([v], [\'lightblue\'])')
print('plt.xlim(-1, 5)')
print('plt.ylim(-1, 5)')
print('plt.show():')
plot_vectors([v], ['lightblue'])
plt.xlim(-1, 5)
_ = plt.ylim(-1, 5)
plt.show()

I = np.array([[1, 0], [0, 1]])
print('\nI:')
print(I)
print('\nIv = np.dot(I, v):')
Iv = np.dot(I, v)
print(Iv)
print('\nv == Iv:')
print(v == Iv)

plot_vectors([Iv], ['blue'])
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.show()

E = np.array([[1, 0], [0, -1]])
print('\nE:')
print(E)
print('\nEv = np.dot(E, v):')
Ev = np.dot(E, v)
print(Ev)

print('\nplot_vectors([v, Ev], [\'lightblue\', \'blue\'])')
print('plt.xlim(-1, 5)')
print('plt.ylim(-3, 3)')
print('plt.show():')
plot_vectors([v, Ev], ['lightblue', 'blue'])
plt.xlim(-1, 5)
plt.ylim(-3, 3)
plt.show()

F = np.array([[-1, 0], [0, 1]])
print('\nF:')
print(F)
print('\nFv = np.dot(F, v):')
Fv = np.dot(F, v)
print(Fv)

print('\nplot_vectors([v, Fv], [\'lightblue\'])')
print('plt.xlim(-4, 4)')
print('plt.ylim(-1, 5)')
print('plt.show():')
plot_vectors([v, Fv], ['lightblue', 'blue'])
plt.xlim(-4, 4)
plt.ylim(-1, 5)
plt.show()

A = np.array([[-1, 4], [2, -2]])
print('\nA:')
print(A)
print('\nAv = np.dot(A, v):')
Av = np.dot(A, v)
print(Av)
print('\nplot_vectors([v, Av], [\'lightblue\', \'blue\'])')
print('plt.xlim(-1, 5)')
print('plt.ylim(-1, 5)')
print('plt.show():')
plot_vectors([v, Av], ['lightblue', 'blue'])
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.show()

v2 = np.array([2, 1])
plot_vectors([v2, np.dot(A, v2)], ['lightgreen', 'green'])
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.show()

v3 = np.array([-3, -1])
v4 = np.array([-1, 1])
# Pada dasarnya vektor hanya memiliki satu dimensi dan tidak memiliki dimensi lain untuk ditransposisikan (dari vektor baris menjadi vektor kolom), oleh karena itu kita akan merubahnya menjadi dua dimensi menggunakan np.matrix() lalu melakukan transposisi pada vektor ini menjadi vektor kolom
print('\nV = np.concatenate((np.matrix(v).T, np.matrix(v2).T, np.matrix(v3).T, np.matrix(v4).T), axis=1)')
V = np.concatenate((np.matrix(v).T,
                    np.matrix(v2).T,
                    np.matrix(v3).T,
                    np.matrix(v4).T),
                   axis=1)
print('V:')
print(V)
print('\nIV = np.dot(I, V):')
print(np.dot(I, V))
print('\nAV = np.dot(A, V):')
AV = np.dot(A, V)
print(AV)
# Fungsi untuk mengkonversi kolom dalam matriks menjadi vektor 1 dimensi


def vectorfy(mtrx, clmn):
    return np.array(mtrx[:, clmn]).reshape(-1)


print('\nvectorfy(V, 0):')
print(vectorfy(V, 0))
print('\nvectorfy(V, 0) == v:')
print(vectorfy(V, 0) == v)
print('\nplot_vectors([vectorfy(V,0), vectorfy(V, 1), vectorfy(V, 2), vectorfy(V, 3), vectorfy(AV, 0), vectorfy(AV, 1), vectorfy(AV, 2), vectorfy(AV, 3)], [\'lightblue\', \'lightgreen\', \'lightgray\', \'orange\', \'blue\', \'green\', \'gray\', \'red\'])')
print('plt.xlim(-4, 6)')
print('plt.ylim(-5, 5)')
print('plt.show():')
plot_vectors([vectorfy(V, 0), vectorfy(V, 1), vectorfy(V, 2), vectorfy(V, 3), vectorfy(AV, 0), vectorfy(AV, 1), vectorfy(
    AV, 2), vectorfy(AV, 3)], ['lightblue', 'lightgreen', 'lightgray', 'orange', 'blue', 'green', 'gray', 'red'])
plt.xlim(-4, 6)
plt.ylim(-5, 5)
plt.show()
