import numpy as np
from encoder import EncoderBlock
from decoder import DecoderBlock
from modules import softmax

class Transformer:
    def __init__(self, vocab_size, d_model=512, d_ff=2048):
        # 1 camada de cada
        self.encoder = EncoderBlock(d_model, d_ff)
        self.decoder = DecoderBlock(d_model, d_ff)

        # Camada linear
        self.W_linear = np.random.randn(d_model, vocab_size) * 0.01
        self.b_linear = np.zeros(vocab_size)
    
    def forward(self, x, y, mask):
        # Passando pela camada de encoder
        Z = self.encoder.forward(x)

        # Passando pela camada de decoder com y, Z e a mascara
        decoder_output = self.decoder.forward(y, Z, mask)

        # Passando pela camada linear
        logits = decoder_output @ self.W_linear + self.b_linear

        # Aplicando softmax para obter as probabilidades
        probabilities = softmax(logits, axis=-1)

        return probabilities