import numpy as np
from dezero import Variable
from dezero.utils import get_dot_graph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 1

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

# 計算グラフのドット言語を出力
gx = x.grad
gx.name = 'gx' + str(iters + 1)
print(get_dot_graph(gx))