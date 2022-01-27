import numpy as np

inicio = []

print(inicio)

A = np.array([1, 2, 3, 1.5, 6, 70, 70.5])

inicio = A
print(inicio)

B = np.array([4, 5, 6, 7, 8, 9, 10])

# C = np.column_stack((A,B))

result = np.argpartition(A, 3)
print(result[:3])

print(B[result[:3]])