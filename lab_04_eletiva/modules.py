import numpy as np

# Funcao de softmax
def softmax(x, axis=-1):
    # Subtrai o valor maximo de cada linha para estabilidade numerica
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x   = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]  # Dimensão dos vetores de chave

    # Calculando os scores de atenção (Q K^T)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1))
    # Dividindo os scores pela raiz quadrada da dimensão dos vetores de chave
    scores = scores / np.sqrt(d_k)

    # Aplicando a mascara
    if mask is not None:
        scores = scores + mask

    # Aplicando softmax para obter as atencoes
    attention_weights = softmax(scores, axis=-1)

    # Multiplicando as atencoes pelos valores (V)
    output = np.matmul(attention_weights, V)

    return output, attention_weights

# Rede FFN

class FeedForwardNetwork:
    def __init__(self, d_model=512, d_ff=2048):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)

        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        # Expansao linear
        ffn_output1 = x @ self.W1 + self.b1
        
        # Ativacao ReLU
        ffn_output1_relu = np.maximum(0, ffn_output1)

        # Contracao linear de volta para d_model
        ffn_output2 = ffn_output1_relu @ self.W2 + self.b2

        return ffn_output2
    
class AddNorm:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def forward(self, x, sublayer_output):
        # Conexao residual (Add)
        x_res  = x + sublayer_output

        # Normalizacao (Norm)
        mean   = np.mean(x_res, axis=-1, keepdims=True)
        var    = np.var(x_res, axis=-1, keepdims=True)
        x_norm = (x_res - mean) / np.sqrt(var + self.epsilon)

        return x_norm
        
        
        
        