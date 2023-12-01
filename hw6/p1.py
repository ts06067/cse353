import numpy as np

SMALLNUMBER = 1e-10


def cross_entropy_loss(y, yhat):
    print("sum: ", -np.sum(y * yhat, axis=0))
    return -np.sum(y * yhat, axis=0) + np.log(np.sum(np.exp(yhat), axis=0))


def cross_entropy_loss_grad(y, yhat):
    return -y + np.exp(yhat) / (np.sum(np.exp(yhat), axis=0) + SMALLNUMBER)


yhat1 = np.array([0.1, 0.25, 0.5, 0.05, 0.1])
yhat2 = np.array([0.1, 0.6, 0.1, 0.1, 0.1])
yhat3 = np.array([0.0, 0.0, 0.125, 0.75, 0.125])
yhat4 = np.array([0.8, 0.1, 0.025, 0.025, 0.05])

y1 = np.array([0, 0, 1, 0, 0])
y2 = np.array([0, 1, 0, 0, 0])
y3 = np.array([0, 0, 0, 1, 0])
y4 = np.array([1, 0, 0, 0, 0])

print("loss:")
print(cross_entropy_loss(y1, yhat1))
print(cross_entropy_loss(y2, yhat2))
print(cross_entropy_loss(y3, yhat3))
print(cross_entropy_loss(y4, yhat4))

print("gradient:")
print(cross_entropy_loss_grad(y1, yhat1))
print(cross_entropy_loss_grad(y2, yhat2))
print(cross_entropy_loss_grad(y3, yhat3))
print(cross_entropy_loss_grad(y4, yhat4))
