import numpy as np

def myLU3(A):
    """ Takes a 3x3 numpy array and computes its
    LU decomposition without partial pivoting
    and assuming that no row swaps are required. """

    L = np.eye(3) # creates indentity matrix to be manipulated by row operations 
    U = A.copy() # sets U to be A 

    multiplier = U[1,0] / U[0,0]
    L[1,0] = multiplier
    U[1] = U[1] - multiplier * U[0]

    multiplier = U[2,0] / U[0,0]
    L[2,0] = multiplier
    U[2] = U[2] - multiplier * U[0]

    multiplier = U[2,1] / U[1,1]
    L[2,1] = multiplier
    U[2] = U[2] - multiplier * U[1]
     
    return L, U

if __name__ == '__main__':
    A = np.array([[1.,2.,-1.],[-2.,-5.,3.],[-1.,-3.,0.]])
    L, U = myLU3(A)
    print(f"Is L in the correct form? {np.allclose(np.triu(L),np.eye(3))}")
    print(f"Is U in the correct form? {np.allclose(U,np.triu(U))}")
    print(f"Do L and U multiply to A? {np.allclose(L @ U, A)}")
