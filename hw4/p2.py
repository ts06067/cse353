import numpy as np


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


X = np.array([[1, 2], [-4, 1], [2, 1], [1, -1], [3, -3], [1, -4]])
y = np.array([-1, -1, -1, 1, 1, 1])

theta_A = np.array([1, -1])
theta_B = np.array([1, -4])
theta_C = np.array([0.01, -0.5])

regression_A = -np.sum(np.log(sigmoid(y * (X @ theta_A))))
regression_B = -np.sum(np.log(sigmoid(y * (X @ theta_B))))
regression_C = -np.sum(np.log(sigmoid(y * (X @ theta_C))))

print("regression_A = ", regression_A)
print("regression_B = ", regression_B)
print("regression_C = ", regression_C)


gradient_A = -np.sum(sigmoid(-y * (X @ theta_A)) * y * X.T, axis=1)
gradient_B = -np.sum(sigmoid(-y * (X @ theta_B)) * y * X.T, axis=1)
gradient_C = -np.sum(sigmoid(-y * (X @ theta_C)) * y * X.T, axis=1)

print("gradient_A = ", gradient_A)
print("gradient_B = ", gradient_B)
print("gradient_C = ", gradient_C)

norm_gradient_A = np.linalg.norm(gradient_A)
norm_gradient_B = np.linalg.norm(gradient_B)
norm_gradient_C = np.linalg.norm(gradient_C)

print("norm_gradient_A = ", norm_gradient_A)
print("norm_gradient_B = ", norm_gradient_B)
print("norm_gradient_C = ", norm_gradient_C)
