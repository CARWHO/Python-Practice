import numpy as np
import matplotlib.pyplot as plt


def forwardSub3(L, b):
    """Given a 3x3 lower triangular 2D array L and a 
       1D array b with length 3, solves Lx = b with
       forward substitution"""
    
    x = np.array([0.0, 0.0, 0.0, 0.0])
    for x_val in range(len(L-1)):
        prev_xs_coeff = 0 #Coefficient of previous x values
        for prev_x in range(0, x_val+1):
            prev_xs_coeff += np.vdot(x[prev_x], L[x_val][prev_x])
        x[x_val] = (b[x_val] - prev_xs_coeff)/L[x_val][x_val]
    return x
    
def backSub3(U, b):
    """Given a 3x3 upper triangular 2D array U and a 
       1D array b with length 3, solves Ux = b with
       back substitution"""
    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    for x_val in range(len(U)-1, -1, -1):
        prev_xs_coeff = 0 #Coefficient of previous x values
        for prev_x in range(x_val, len(x)):
            prev_xs_coeff += np.vdot(x[prev_x], U[x_val][prev_x])
        x[x_val] = (b[x_val] - prev_xs_coeff) / U[x_val][x_val]
    
    return x

def myLU3(A):
    """Takes an array and returns the LU decompesition of that matrix"""
    U = A.copy()
    n = len(U)
    L = np.identity(n)
    for i in range(n):
        pivot = U[i,i]
        for next_row in range(i+1, n):
            coeff = U[next_row, i] / pivot #Coefficient required to eliminate the below term
            U[next_row] -= coeff * U[i]
            L[next_row, i] += coeff
    return L, U 

def solve_power3(L, U, k, b):
    x = b
    for i in range(k):
        y = forwardSub3(L, x)
        x = backSub3(U, y)
    return x

def shifted_inverse_power_method(A, shift, max_iter = 100, tol=1e-8):
    n = A.shape[0]
    I = np.eye(n)
    A_shift = A - shift * I
    x = np.ones(n)
    x = x/np.linalg.norm(x, np.inf)
    r_old = np.nan
    L, U = myLU3(A)
    for i in range(max_iter):   
        y = solve_power3(L, U, 1, x)
        mu = np.dot(x, y)
        new_x = y/np.linalg.norm(y, np.inf)
        r = y.dot(x)/x.dot(x)
        x = y/np.linalg.norm(y, np.inf)
        err = np.abs(r - r_old)/np.abs(r)
        r_old = r
        if err < tol:
            break
        x = new_x
    eig_val = 1/mu + shift
    return eig_val

def calculate_eig_vals(A, shifts, max_iter = 100, tol = 1e-8,):
    eig_vals = np.array([])
    for shift in shifts:
        eig_val = shifted_inverse_power_method(A, shift, max_iter, tol)
        eig_vals = np.append(eig_vals, eig_val)
    return eig_vals

def gershgorin_disc_display(A, centres, row_radii, col_radii):
    fig, ax = plt.subplots()
    ax.grid()
    for i in range(A.shape[0]):
        ax.add_patch(plt.Circle((centres[i].real, centres[i].imag), row_radii[i], color='blue', alpha=.2))
        ax.add_patch(plt.Circle((centres[i].real, centres[i].imag),col_radii[i], color='green', alpha=.2))
    ax.axis('equal')
    plt.show()

def eigenvalues_display(A, centres):
    fig, ax = plt.subplots()
    ax.grid()
    ax.axis('equal')
    eigenvalues = calculate_eig_vals(A, centres)
    ax.plot(eigenvalues.real, eigenvalues.imag, 'ro')
    plt.show()


def main():
    A = np.array([[5, 0.1j, -1+0.5j, 0.1j], [0.1, 2+0.5j, 0.1+0.1j, 0], [-0.1j, 0.1, 5, 2], [0, 1, 0.5, -3.5-1j]])

    centres = [5, 2 + 0.5j, 5, -3.5 - 1j]
    row_radii = [-1 + 0.7j, 0.2 + 0.1j, 2.1 - 0.1j, 1.5]
    col_radii = [0.1 - 0.1j, 1.1 + 0.1j, -0.4 + 0.6j, 2 + 0.1j]

    gershgorin_disc_display(A, centres, row_radii, col_radii)
    eigenvalues_display(A, centres)

main()