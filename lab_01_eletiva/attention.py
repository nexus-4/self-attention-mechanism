import numpy as np

def softmax(x, axis=-1):
    """
    Calcula o softmax de cada linha da matriz x.
    """
    # Subtrai o valor máximo de cada linha para estabilidade numérica
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class ScaledDotProductAttention:
    def __init__(self):
        pass

    def forward(self, Q, K, V):
        """
        Calcula a atenção dadas as matrizes de consulta (Q), chave (K) e valor (V).
        """
        
        # Produto Escalar (Q K^T)
        scores = np.matmul(Q, K.T) 

        # Dimensao dos vetores de chave
        d_k = K.shape[-1]

        # Aplicando o scaling factor (sqrt(d_k))
        scaled_scores = scores / np.sqrt(d_k)

        # Aplicando softmax para obter as atenções linha a linha
        attention_weights = softmax(scaled_scores)

        # Multiplicando as atenções pelos valores (V)
        output = np.matmul(attention_weights, V)

        return output, attention_weights

