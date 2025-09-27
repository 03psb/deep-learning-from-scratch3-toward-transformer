import numpy as np
import dezero
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

        self.W_Q = Linear(d_k, nobias=True, dtype=dtype, in_size=d_model)
        self.W_K = Linear(d_k, nobias=True, dtype=dtype, in_size=d_model)
        self.W_V = Linear(d_v, nobias=True, dtype=dtype, in_size=d_model)
        
    def forward(self, x):
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        s_out = F.softmax(F.matmul(q, k.T) / np.sqrt(d_model))
        y = F.matmul(s_out, v)
        return y


# =============================================================================
# Multi Head Attention
# =============================================================================
class MultiHeadAttention(Layer):
    def __init__(self, d_model, d_k, d_v, h, dtype=np.float32):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dtype = dtype
        self.h = h
        self.heads = []
        
        for _ in range(h):
            head = SingleHeadAttention(d_model, d_k, d_v, dtype)
            heads.append(head)

        self.W_O = Linear(d_model, nobias=True, dtype=dtype, in_size=h * d_v)
            
    def forward(self, x):
        heads = Parameter(np.zeros(x.shape[0], h * d_v), dtype=dtype)
        
        for i in range(self.h):
            h_out = heads[i].forward(x)
            heads[i * d_v: (i + 1) * d_v][...] = h_out
        
        y = F.matmul(heads, self.W_O)
        return y
