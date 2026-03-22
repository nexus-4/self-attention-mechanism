import numpy as np
from modules import scaled_dot_product_attention, FeedForwardNetwork, AddNorm

class EncoderBlock:
    def __init__(self, d_model=512, d_ff=2048):
        # Matrizes de pesos
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01

        # Subcamadas
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        # Duas camadas de Add & Norm
        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()

    def forward(self, x):
        # Gerando Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Chamndo a funcao scaled_dot_product_attention
        attention_output, _ = scaled_dot_product_attention(Q, K, V)

        # Passando pela primeira camada de Add & Norm
        x_norm1 = self.add_norm1.forward(x, attention_output)

        # Passando pela rede FFN
        ffn_output = self.ffn.forward(x_norm1)

        # Passando pela segunda camada de Add & Norm
        x_norm2 = self.add_norm2.forward(x_norm1, ffn_output)

        return x_norm2