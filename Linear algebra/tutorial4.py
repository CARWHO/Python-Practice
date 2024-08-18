import numpy as np
from tutorial3 import myLU3  # From last week!
# You’ll have to change "EMTH211_Tutorial_3_Code" to whatever you named the file
# and put the file into the same folder.
# Alternatively, you could just copy the function manually.

def forwardSub3(L, b):
    """Given a 3x3 lower triangular 2D array L and a 
       1D array b with length 3, solves Lx = b with
       forward substitution"""
    
    x_1 = b[0] / L[0, 0]
    x_2 = (b[1] - L[1, 0] * x_1) / L[1, 1]
    x_3 = (b[2] - L[2, 0] * x_1 - L[2, 1] * x_2) / L[2, 2]

    return np.array([x_1, x_2, x_3])

def backSub3(U, b):
    """Given a 3x3 upper triangular 2D array U and a 
       1D array b with length 3, solves Ux = b with
       back substitution"""
    
    x_3 = b[2] / U[2, 2]
    x_2 = (b[1] - U[1, 2] * x_3) / U[1, 1]
    x_1 = (b[0] - U[0, 1] * x_2 - U[0, 2] * x_3) / U[0, 0]

    return np.array([x_1, x_2, x_3])

if __name__ == '__main__':
    L = np.array([[1.,0.,0.],[2.,1.,0.],[2.,1.,1.]])
    U = np.array([[1.,2.,3.],[0.,4.,5.],[0.,0.,6.]])
    b = np.array([2.,1.,1.])

    x1 = forwardSub3(L,b)
    x2 = backSub3(U,b)

    print(f"The solution to Lx = b is x = {x1}")
    print(f"The solution to Ux = b is x = {x2}")
