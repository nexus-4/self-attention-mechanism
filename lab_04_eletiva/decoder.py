import numpy as np
from modules import scaled_dot_product_attention, FeedForwardNetwork, AddNorm

class DecoderBlock:
    def __init__(self, d_model=512, d_ff=2048):
        # Pesos para masked self-attention
        self.W_q1 = np.random.randn(d_model, d_model) * 0.01
        self.W_k1 = np.random.randn(d_model, d_model) * 0.01
        self.W_v1 = np.random.randn(d_model, d_model) * 0.01

        # Pesos para cross-attention
        self.W_q2 = np.random.randn(d_model, d_model) * 0.01
        self.W_k2 = np.random.randn(d_model, d_model) * 0.01
        self.W_v2 = np.random.randn(d_model, d_model) * 0.01

        # FFN e 3 camadas de Add & Norm
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()
        self.add_norm3 = AddNorm()

    def forward(self, y, Z, mask):
        # Masked self-attention
        Q1 = y @ self.W_q1
        K1 = y @ self.W_k1
        V1 = y @ self.W_v1

        # Passando pela funcao scaled_dot_product_attention com a mascara
        self_att_out, _ = scaled_dot_product_attention(Q1, K1, V1, mask=mask)

        # Passando pela primeira camada de Add & Norm
        y_norm1 = self.add_norm1.forward(y, self_att_out)

        # Cross-attention
        Q2 = y_norm1 @ self.W_q2
        K2 = Z @ self.W_k2
        V2 = Z @ self.W_v2
        cross_att_out, _ = scaled_dot_product_attention(Q2, K2, V2)

        # Passando pela segunda camada de Add & Norm
        y_norm2 = self.add_norm2.forward(y_norm1, cross_att_out)

        # Passando pela rede FFN
        ffn_out = self.ffn.forward(y_norm2)

        # Passando pela terceira camada de Add & Norm
        y_norm3 = self.add_norm3.forward(y_norm2, ffn_out)

        return y_norm3