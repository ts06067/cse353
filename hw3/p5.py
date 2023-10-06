import numpy as np

A = np.array([[1, 2, 0], [-2, 3, 4]])
yA = np.array([1, -1])
thetaA = np.array([1, 2, 3])

B = np.array([[1, 1], [1, 1], [1, 1]])
yB = np.array([10, 20, 30])
thetaB = np.array([1, 2])

a = 0.5

coeff_list = [A, B]
y_list = [yA, yB]
theta_list = [thetaA, thetaB]

for i in range(len(coeff_list)):
    try:
        new_theta = theta_list[i] - a * coeff_list[i].T @ (
            coeff_list[i] @ theta_list[i] - y_list[i]
        )
        print("new_theta = ", new_theta)
    except:
        print("no unique solution for coeff_list[", i, "]")
