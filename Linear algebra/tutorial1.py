# section 2

#section 2.1

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# section 2.2

# A = np.array ([[16 , 3 , 2 , 13], [5 , 10 , 11 , 8], [9 , 6 , 7 , 12], [4 , 15 , 14 , 1]])
#print(A)

# section 2.3

# N/A

# section 2.4

#B = A.sum(axis =0)
#C = A.sum(axis =1)
#D = A.transpose()
#E = A.T

#if E.all() == D.all():
    #print("Success")
    #print(D)

# section 2.5

#A[0 , 3] + A[1 , 3] + A[2 , 3] + A[3 , 3]
#t = A[3 , 4]
#print(t)

# section 2.6

# print(np.linspace(0, np.pi, 22)) 
# print(np.arange(10, 50, 0.1))

# section 2.7

#print(A[0:4 , 3].sum())
#print(A[:, -1].sum())

# section 3

#section 3.1

#B = np.array ([[3 , 4], [5 , 6]])
#print(B)
#I = (np.eye(2))
#print(I)

# seciton 3.2

# print(B@I)

# section 4

# section 4 & 4.1

# C = np.array([[1 , 2],
              # [3 , 4]])
# D = np.array([5,6])
# print(np.row_stack((C,D)))
# print(np.column_stack((C,D)))

#section 4.2

# N/A

#section 4.3

# i = 0

#print(A)
# A[i-1] *= C
# A[i-1] += C * A[j-1]
# print(A)

#section 5

# print(A.T@A)
# D = la.det(A)
# print(D)

#section 5.1 

A = np.array([[1 , 1 , 1],
              [1 , 2 , 3],
              [1 , 3 , 6]])
b = np.array([3 , 1 , 4])
x = la.solve(A , b)
print(x)


#section 6 



