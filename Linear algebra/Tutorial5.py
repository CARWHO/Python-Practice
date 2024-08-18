import numpy as np
from tutorial3 import myLU3
from tutorial4 import forwardSub3, backSub3

def solve_power3(A, k, b):
    """Solves the system A**k @ x = b for 3x3 A"""
    A_k = np.eye(3)  
    for _ in range(k):
        A_k = A_k @ A  
    L, U = myLU3(A_k)  
    y = forwardSub3(L, b)  
    x = backSub3(U, y)  
    return x


if __name__ == "__main__":
    A = np.array([[1,0,1],[0,1,0],[0,1,1]],dtype=float)
    b = np.ones(3)
    k = 5    
    x = solve_power3(A, k, b)
    print(f"{x = }")
    # A @ A @ A @ A @ A is bad, but the question prohibits np.linalg.matrix_power
    print(f"Does x solve this system? {np.allclose((A @ A @ A @ A @ A) @ x, b)}")