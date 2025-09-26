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

        self.w_q = Linear(d_k, in_size=d_model)
        self.w_k = Linear(d_k, in_size=d_model)
        self.w_v = Linear(d_v, in_size=d_model)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        attention = F.softmax(F.matmul(q, k.T) / )
        
