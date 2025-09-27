import numpy as np
import dezero
import dezero.functions as F
from dezero import Variable
from dezero.layers import Layer
from dezero.transformers import MultiHeadAttention


# =============================================================================
# Encoder Block
# =============================================================================
class Encoder(Layer):
    def __init__(self, d_model, d_k, d_v, h, d_ff, dtype=np.float32):
        pass


# =============================================================================
# Decoder Block
# =============================================================================
class Decoder(Layer):
    def __init__(self, d_model, d_k, d_v, h, d_ff, dtype=np.float32):
        pass


# =============================================================================
# Embedding
# =============================================================================
class Embedding(Layer):
    def __init__(self):
        pass


# =============================================================================
# Positional Encoding
# =============================================================================
class PosEncoding(Layer):
    def __init__(self):
        pass
