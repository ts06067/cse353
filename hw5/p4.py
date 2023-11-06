import numpy as np

theta = np.array([[1, -1, 500], [4, 1, 1]])

# normalize theta by dividing each column by its norm
theta_norm = theta / np.linalg.norm(theta, axis=0)

x = np.array([[4, 4, 7, -4, 3], [5, -3, -6, 0, 3]])

print(np.matmul(theta_norm.T, x))
