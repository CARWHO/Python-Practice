import numpy as np

def power_method(A, x0, max_iter=100, tol=1e-6):
    """Finds an eigenvector associated with the dominant eigenvalue."""

    x = x0

    for _ in range(max_iter):
        Ax = A @ x
        x_new = Ax / np.linalg.norm(Ax, np.inf)
        if np.linalg.norm(x_new, np.inf) < tol:
            break
        x = x_new
    
    return x

if __name__ == "__main__":
    n = 3
    np.random.seed(seed=211)
    A = np.random.rand(n, n)
    x0 = np.ones(n)
    x = power_method(A, x0, max_iter=100)
    Ax = A @ x
    R = (Ax).dot(x) / x.dot(x)  # One last Rayleigh quotient for our most recent x
    print(f"{x = }")
    print(f"A @ x = {Ax}")
    print(f"lambda * x = {R * x}")
