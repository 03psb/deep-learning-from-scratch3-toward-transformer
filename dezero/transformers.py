import dezero.functions as F
from dezero import cuda
from dezero.core import Parameter



# =============================================================================
# Single Head Attention
# =============================================================================
class SingleHeadAttention(Layer):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
