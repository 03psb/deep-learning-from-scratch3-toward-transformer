import numpy as np
# core_simple を明示的にインポート
# （dezero/__init__.py の is_simple_core = False でも動作させるため）
from dezero.core_simple import Variable
from dezero.core_simple import setup_variable
setup_variable()


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


logs = []
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
iters = 1000
lr = 0.001

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    logs.append([float(x0.data), float(x1.data), float(y.data)])

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

# 図を描画
import matplotlib.pyplot as plt

R = 0.01
x = np.arange(-2.0, 2.0, R)
y = np.arange(-1.0, 3.0, R)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
XX = [float(d[0]) for d in logs]
YY = [float(d[1]) for d in logs]
ZZ = [float(d[2]) for d in logs]
plt.plot(XX, YY, c='orange', alpha=0.8)
plt.scatter(XX, YY, c='red', alpha=0.5)
plt.scatter([1], [1],marker="*",s=100,linewidths="2", c='blue')
plt.contour(X, Y, Z, alpha=0.5, levels=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
plt.show()