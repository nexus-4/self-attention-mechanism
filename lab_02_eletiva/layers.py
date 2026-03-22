import numpy as np
from data_preparation import X

class EncoderLayer:
    def __init__(self,d_model=64):
        self.d_model = d_model
        self.W_q     = np.random.randn(d_model,d_model)
        self.W_k     = np.random.randn(d_model,d_model)
        self.W_v     = np.random.randn(d_model,d_model)
        self.W1      = np.random.randn(d_model, d_model * 4)
        self.b1      = np.random.randn(d_model * 4)
        self.W2      = np.random.randn(d_model * 4, d_model)
        self.b2      = np.random.randn(d_model)
    
    def forward(self, X):
        # Gerando as matrizes de consulta, chave e valor
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # O produto escalar (scores)
        scores = Q @ np.swapaxes(K, -1, -2)

        # Multiplicando os scores pela raiz quadrada da dimensão do modelo
        scores_scaled = scores / np.sqrt(self.d_model)

        # softmax para obter as probabilidades de atenção
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)
        
        attention_weights = softmax(scores_scaled)

        # Calculando a saída da atenção
        output = attention_weights @ V

        X_res  = X + output

        # Calculando a media e variancia
        mean = np.mean(X_res, axis=-1, keepdims=True)
        var  = np.var(X_res, axis=-1, keepdims=True)

        # Layer Normalization
        epsilon = 1e-6
        X_norm  = (X_res - mean) / np.sqrt(var + epsilon)
        
        # Feed Forward Network
        ffn_output1       = X_norm @ self.W1 + self.b1
        ffn_output1_relu  = np.maximum(0, ffn_output1)
        ffn_output2       = ffn_output1_relu @ self.W2 + self.b2
        
        X_res2 = X_norm + ffn_output2
        
        # Segunda normalização
        mean2 = np.mean(X_res2, axis=-1, keepdims=True)
        var2  = np.var(X_res2, axis=-1, keepdims=True)
        X_out = (X_res2 - mean2) / np.sqrt(var2 + epsilon)

        return X_out

        
    
class TransformerEncoder:
    def __init__(self, num_layers=6, d_model=64):
        self.layers = [EncoderLayer(d_model) for _ in range(num_layers)]

    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)

        return X
