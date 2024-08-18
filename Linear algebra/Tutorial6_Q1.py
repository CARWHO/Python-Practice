import numpy as np
n = 3
np.random.seed(seed=211)
A = np.random.rand(n, n)

vals, vecs = np.linalg.eig(A)

print("Eigenvalues:")
print(vals)

print("Eigenvectors:")
print(vecs)
