import numpy as np

A = np.array([[1, 2], [3, 4]])
yA = np.array([5, 6])

B = np.array([[1, 0], [0, 2]])
yB = np.array([56, 512])

C = np.array([-5, -2])
yC = np.array([-14])

D = np.array([[1, -1], [-1, 1]])
yD = np.array([5, 10])

E = np.array([[1, 1], [1, 1], [1, 1]])
yE = np.array([10, 20, 30])

F = np.array([1, 2, 3, 4, 5])
yF = np.array([10, 100, 1000, 10000, 100000])

coeff_list = [A, B, C, D, E, F]
y_list = [yA, yB, yC, yD, yE, yF]

for i in range(len(coeff_list)):
    try:
        print("coeff_list[", i, "] = ", coeff_list[i])
        print("y_list[", i, "] = ", y_list[i])
        print(
            "np.linalg.solve(coeff_list[",
            i,
            "], y_list[",
            i,
            "]) = ",
            np.linalg.solve(coeff_list[i], y_list[i]),
        )
    except:
        print("no unique solution for coeff_list[", i, "]")
