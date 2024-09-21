import numpy as np

def jacobi(A, b, x0, max_iter=10):
    # Set up anything you need

    results = [] ## final matrix
    x = x0.copy() ## copy of initial guess 

    for k in range(max_iter):
        # Do the iterative method   

        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i]) ## ith row of A
            s2 = np.dot(A[i, i + 1:], x[i + 1:]) ## ith row of A, after ith element (i.e. i + 1)
            x_new[i] = (b[i] - s1 - s2) / A[i, i] 
        x = x_new ## restarts 
        results.append([k + 1, *x])
    return np.array(results)

if __name__ == "__main__":
    A = np.array([[2, 0.5, -0.5], [0, 4, -2], [4, 0, -4]])
    b = np.array([4., 4., 0.])
    x0 = np.array([1., 0., 0.])
    print(jacobi(A, b, x0))
