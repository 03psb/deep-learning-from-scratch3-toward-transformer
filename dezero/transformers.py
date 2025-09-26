import numpy as np
import dezero.functions as F
from dezero import cuda
from dezero.core import Parameter
from dezero.layers import Linear



# =============================================================================
# Single Head Attention
# =============================================================================
class SingleHeadAttention(Layer):
    def __init__(self, d_model, d_k, d_v, dtype=np.float32):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dtype = dtype

        self.W_Q = Linear(d_k, in_size=d_model)
        self.W_K = Linear(d_k, in_size=d_model)
        self.W_V = Linear(d_v, in_size=d_model)
        
    def forward(self, x):
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        s_out = F.softmax(F.matmul(q, k.T) / np.sqrt(d_model))
        y = F.matmul(s_out, v)
        return y
