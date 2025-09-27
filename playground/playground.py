import numpy as np
from dezero import Variable, Model
from dezero.transformers import SingleHeadAttention
import dezero.functions as F


class SingleHeadAttentionNet(Model):
    def __init__(self, d_model, d_k, d_v, dtype=np.float32):
        super().__init__()
        self.l1 = SingleHeadAttention(d_model, d_k, d_v, dtype)


    def forward(self, x):
        y = self.l1(x)
        return y

d_model = 512
d_k = 64
d_v = 64
x = Variable(np.random.randn(5, 512), name='x')
model = SingleHeadAttentionNet(d_model, d_k, d_v)
model.plot(x)
