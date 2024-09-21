import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

def bigO(n, time_taken):
    if time_taken > 0:
        flops = 2/3 * (n**3)
        flop_per_time = flops/time_taken
        print(f"Total flops = {flops}")
        print(f"{flop_per_time}/(s)")
    else:
        print("Matrix too small to measure")

# cubic function with a constant offset
def cubic(n, a, b):
    return a * (n ** 3) + b

# provided code starts here # ---------------------------------------------------------------------------

times = []
sizes = list(range(2, 5001, 50))

for n in sizes:
    A = np.random.rand(n, n)  # Make a random matrix
    t0 = time.time()  # Start time
    np.linalg.inv(A)
    t1 = time.time()  # Stop time
    time_taken = t1 - t0
    times.append(t1 - t0)  # Time taken, in seconds
    bigO(n, time_taken)

# provided code ends here # -----------------------------------------------------------------------------

start_n = sizes[0]
end_n = sizes[-1]
start_time = times[0]
end_time = times[-1]

def fit_with_boundary_conditions(n, a):
    b = (start_time - a * (start_n ** 3))  # solving for b using the start point
    return cubic(n, a, b)

popt, _ = curve_fit(fit_with_boundary_conditions, sizes, times) 

a_fitted = popt[0]
b_fitted = start_time - a_fitted * (start_n ** 3)  # calculate b using the start point
theoretical_times = [a_fitted * (n**3) + b_fitted for n in sizes] # theoretical curve based on flop count

# Plotting the measured data and fitted curve
plt.plot(sizes, times, marker='o', label='Measured Data')
plt.plot(sizes, cubic(np.array(sizes), a_fitted, b_fitted), '-', label=f'Fitted Curve: O(n^3) with boundary conditions')
plt.xlabel('Matrix(n)')
plt.ylabel('Time(s)')
plt.title("Time Taken To Invert Matrices")
plt.legend()
plt.grid(True)
plt.show()
