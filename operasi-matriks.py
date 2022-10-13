import numpy as np
import matplotlib.pyplot as plt
import torch

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

print('\n---EIGENVECTORS AND EIGENVALUES')
print('A:')
print(A)
print('\nMenggunakan eig() pada NumPy akan mengembalikan tuple yang mengandung vektor dari eigenvalues dan matriks eigenvektor')
print('lambdas, v = np.linalg.eig(A):')
lambdas, V = np.linalg.eig(A)
print('V:')
print(V)
print('lambdas:')
print(lambdas)
v = V[:, 0]
print('\nv:')
print(v)
lambduh = lambdas[0]
print('\nlambduh: #lambdas direservasi oleh sistem python, sehingga kita menggunakan lambduh untuk mencegah error')
print(lambduh)
print('\nAv = np.dot(A, v):')
Av = np.dot(A, v)
print(Av)
print('\nlambduh * v:')
print(lambduh * v)
print('\nplot_vectors([Av, v], [\'blue\', \'lightblue\'])')
print('plt.xlim(-1, 2)')
print('plt.ylim(-1, 2)')
print('plt.show():')
plot_vectors([Av, v], ['blue', 'lightblue'])
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.show()
v2 = V[:, 1]
lambda2 = lambdas[1]
print('\nAv2 = np.dot(A, v2):')
Av2 = np.dot(A, v2)
print(Av2)
print('\nlambda2 * v2:')
print(lambda2 * v2)
print(
    '\nplot_vectors([Av, v, Av2, v2], [\'blue\', \'lightblue\', \'green\', \'lightgreen\'])')
print('plt.xlim(-1, 4)')
print('plt.ylim(-3, 2)')
print('plt.show():')
plot_vectors([Av, v, Av2, v2], ['blue', 'lightblue', 'green', 'lightgreen'])
plt.xlim(-1, 4)
plt.ylim(-3, 2)
plt.show()
A_p = torch.tensor([[-1, 4], [2, -2.]], dtype=torch.cfloat)
print('\nA_p:')
print(A_p)
print('\nlambdas_p, V_p = torch.linalg.eig(A_p)')
lambdas_p, V_p = torch.linalg.eig(A_p)
v_p = V_p[:, 0]
print('v_p = V_p[:, 0]:')
print(v_p)
print('\nlambda_p = lambdas_p[0]:')
lambda_p = lambdas_p[0]
print(lambda_p)
print('\nAv_p = torch.matmul(A_p, v_p):')
Av_p = A_p @ v_p
print(Av_p)
print('\nlambda_p * v_p:')
print(lambda_p * v_p)
v2_p = V_p[:, 1]
print('\nv2_p = V_p[:, 1]:')
print(v2_p)
print('\nlambda2_p = lambdas_p[1]:')
lambda2_p = lambdas_p[1]
print(lambda2_p)
print('\nAv2_p = torch.matmul(A_p, v2_p):')
Av2_p = torch.matmul(A_p, v2_p)
print(Av2_p)
print('\nlambda2_p * v2_p:')
print(lambda2_p * v2_p)
print(
    '\nplot_vectors([Av_p.numpy(), v_p.numpy(), Av2_p.numpy(), v2_p.numpy()], [\'blue\', \'lightblue\', \'green\', \'lightgreen\'])')
print('plt.xlim(-1, 4)')
print('plt.ylim(-3, 2)')
print('plt.show():')
plot_vectors([np.asarray(Av_p, dtype='float'), np.asarray(v_p, dtype='float'), np.asarray(Av2_p, dtype='float'), np.asarray(v2_p, dtype='float')], [
             'blue', 'lightblue', 'green', 'lightgreen'])
plt.xlim(-4, 3)
plt.ylim(-2, 3)
plt.show()

X = np.array([[25, 2, 9], [5, 26, -5], [3, 7, -1]])
print('\nX:')
print(X)
lambdas_X, V_X = np.linalg.eig(X)
print('\nV_X:')
print(V_X)
print('\nlambdas_X:')
print(lambdas_X)
print('Confirm v_X * X = lambdas_X * v_X:')
v_X = V_X[:, 0]
print('v_X:')
print(v_X)
lambda_X = lambdas_X[0]
print('lambda_X:')
print(lambda_X)
print('\nnp.dot(X, v_X):')
print(np.dot(X, v_X))
print('\nlambda_X * v_X:')
print(lambda_X * v_X)

print('\n---MATRIX DETERMINANTS')
X = np.array([[4, 2], [-5, -3]])
print('\nX:')
print(X)
print('\nnp.linalg.det(X):')
print(np.linalg.det(X))

X = np.array([[1, 2, 4], [2, -1, 3], [0, 5, 1]])
print('\nX:')
print(X)
print('\nnp.linalg.det(X):')
print(np.linalg.det(X))


X = np.array([[1, 2, 4], [2, -1, 3], [0, 5, 1]])
print('\n---DETERMINANT AND EIGENVALUES:')
print('\nX:')
print(X)
print('\nnp.linalg.det(X):')
print(np.linalg.det(X))
lambdas, V = np.linalg.eig(X)
print('\nlambdas:')
print(lambdas)
print('\nnp.product(lambdas):')
print(np.product(lambdas))
