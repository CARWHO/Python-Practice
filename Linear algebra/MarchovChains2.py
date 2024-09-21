import numpy as np
from fractions import Fraction
from tabulate import tabulate 

T = np.zeros((10, 10))

loop_back_prob_a = 5.5/6        # loopback probability for node A
loop_back_prob_others = 1/6   # loopback probability for all other node
transition_prob = 2.5/6         # probability of moving to a neighbour node

connections = {
    0: [1],             # A -> B
    1: [0, 3, 2, 8, 9], # B -> A, D, C, I, J
    2: [3, 5, 1, 8],    # C -> D, F, B, I
    3: [1, 2, 4],       # D -> B, C, E
    4: [3, 5],          # E -> D, F
    5: [6, 7, 8, 2, 4], # F -> G, H, I, C, E
    6: [5, 7],          # G -> F, H
    7: [5, 8, 9, 6],    # H -> F, I, J, G
    8: [9, 7, 5, 2, 1], # I -> J, H, F, C, B
    9: [7, 8, 1]        # J -> H, I, B
}

for node, neighbors in connections.items():
    if node == 0:
        T[node, node] = loop_back_prob_a  # node a loop-back probability
    else:
        T[node, node] = loop_back_prob_others  # loop back prob for other nodes
    
    for neighbor in neighbors:
        T[node, neighbor] = transition_prob  # Probability of moving to each neighboring node
    
    T[node, :] /= T[node, :].sum()  # Normalize

# Transpose for display
T_transposed_for_display = T.T
T_fractions = [[Fraction(val).limit_denominator() for val in row] for row in T_transposed_for_display]
T_list = [[str(fraction) for fraction in row] for row in T_fractions]

print("Transition Matrix (Transposed):")
print(tabulate(T_list, tablefmt="fancy_grid"))

# Solve for steady-state vector
eigenvalues, eigenvectors = np.linalg.eig(T.T)
steady_state_vector = eigenvectors[:,0]
steady_state_vector = steady_state_vector / np.linalg.norm(steady_state_vector, ord=1)  # Normalize eigenvector

print("Steady-State Vector:")
for value in steady_state_vector:
    print(round(-value * 100, 4))
