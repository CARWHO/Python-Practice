import numpy as np
import matplotlib.pyplot as plt

def linearFit(data):
    # Do the least squares thing
    A = data

    a_1 = data[:, 0]
    a_2 = data[:, 1]

    v1 = a_1
    q_1 = (1 / np.linalg.norm(a_1)) * a_1

    v2 = a_2 - (np.dot(a_2, q_1) * q_1)
    q_2 = (1 / np.linalg.norm(v2)) * v2  

    Q = np.column_stack((q_1, q_2))
    R = np.dot(Q.T, data)

    c = np.linalg.solve(R, np.dot(Q.T, data[:, 1]))
    y_fit = np.dot(Q, c)
    err = np.linalg.norm(np.dot(Q, c) - a_2)
    
    return c, err, y_fit

if __name__ == "__main__":
    data = np.genfromtxt("data_1.csv", delimiter=",")
    c, err, y_fit = linearFit(data)
    print(f"The linear model is y ~ {c[0]:.2g} + ({c[1]:.2g})x with a least squares error of {err:.4f}")
    # Do the plotting thing

    plt.scatter(data[:, 0], data[:, 1], label='Data points')
    plt.plot(data[:, 0], y_fit, color='red', label='Fitted line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Fit')
    plt.legend()
    plt.show()