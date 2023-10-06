import numpy as np

t = np.array([0, 0.1, 0.5, 1, 1.1, 2, 10])
v = np.array([2.5, 1.5, -2.2, -7.3, -8.1, -17.0, -95.4])
m = 7

print("t = ", t)
print("v = ", v)

x = np.array([t, np.ones(m)]) @ v * 2 / m
y = np.array([t, np.ones(m)]) @ np.array([t, np.ones(m)]).T * 2 / m

print("x = ", x)
print("y = ", y)

theta = np.linalg.solve(y, x)
print("theta = ", theta)

mse = np.linalg.norm(v - np.array([t, np.ones(m)]).T @ theta, 2) ** 2 / m

print("mse = ", mse)
