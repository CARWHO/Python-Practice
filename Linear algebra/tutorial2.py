import numpy as np

def myRowEchelon3(A):
    # Takes a 3x3 matrix and uses Gaussian
    # elimination to find its row echelon form .
    # Your code here 

    A[1,0] = A[1,0] - ((A[1,0])/(A[0,0]))*A[0,0]
    A[2,0] = A[2,0] - ((A[2,0])/(A[0,0]))*A[0,0]
    A[2,1] = A[2,1] - ((A[2,1])/(A[0,0]))*A[0,0]

    return A

if __name__ == '__main__': 
    np.random.seed(seed=211)  # Sets the seed so everyone gets the same matrix
    A = np.array([[2., -1., 1.], [-2., 3., -1.], [4., -15., 7.]])
    B = np.random.rand(3, 3)  # A random matrix
    C = np.array([[1000000., 100., 1000000.], [200., -100., 100.], [0., 0., 0.]])

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nMatrix C:")
    print(C)

    print("\nRow Echelon Form of A:")
    print(myRowEchelon3(A))
    print("\nRow Echelon Form of B:")
    print(myRowEchelon3(B))
    print("\nRow Echelon Form of C:")
    print(myRowEchelon3(C))
