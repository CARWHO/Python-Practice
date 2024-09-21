import numpy as np

def steady_state(T):
    n = T.shape[0]
    T_transposed = T.T
    T_minus_eye = T_transposed - np.eye(n)
    b = np.zeros(n)
    b[-1] = 1 
    T_minus_eye[-1, :] = np.ones(n)
    T_reduced = np.linalg.solve(T_minus_eye, b)
    
    return T_reduced

T = np.zeros((10, 10))
loop_back_prob = 0.05 #small chance to repeat
transition_prob = 5/6

#original connections

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

#improved att1

# connections = {
#     0: [1, 3],   # A -> B, D
#     1: [0, 8],   # B -> A, I
#     2: [3, 5],   # C -> D, F
#     3: [2, 0],   # D -> C, A //
#     4: [5, 7],   # F -> C, E
#     5: [4, 6],   # E -> F, H //
#     6: [5, 7],   # H -> E, G // 
#     7: [6, 9],   # G -> H, J // 
#     8: [9, 1],   # I -> J, B // 
#     9: [8, 7]    # J -> I, G // 
# }

#improved, more even distribution 

# connections = {
#     0: [1, 3],   # A -> B, D
#     1: [0, 8],   # B -> A, I
#     2: [3, 5],   # C -> D, F
#     3: [0, 2],   # D -> A, C
#     4: [5, 7],   # E -> F, H
#     5: [2, 4],   # F -> C, E
#     6: [7, 9],   # G -> H, J
#     7: [4, 6],   # H -> E, G
#     8: [1, 9],   # I -> B, J
#     9: [6, 8]    # J -> G, I
# }

for node, neighbors in connections.items():
    T[node, node] = loop_back_prob  
    for neighbor in neighbors:
        T[node, neighbor] = transition_prob
    
    T[node, :] /= T[node, :].sum() #normalise

Ss = steady_state(T)
print(Ss)


Ss = steady_state(T)
print(T)
print()
for prob in Ss:
    prob_perc = prob * 100
    print(f"{prob_perc:.2f}%")
print()


