import numpy as np


class Variable:

    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 関数を取得
            x, y = f.input, f.output  # 関数の入出力を取得
            x.grad = f.backward(y.grad)  # backwardを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator)


class Function:

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # 親を覚えさせる
        self.input = input
        self.output = output  # 出力を覚える
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy


class Exp(Function):

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)  # 3.297442541400256