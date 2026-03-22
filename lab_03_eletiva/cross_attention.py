import numpy as np

# A ponte Encoder-Decoder (Cross-Attention)

# Definindo os parametros primeiro
d_model = 512
d_k     = 64
d_v     = 64

# Tensores ficticios de entrada do encoder e do decoder 
encoder_output = np.random.randn(1, 10, d_model)
decoder_state  = np.random.randn(1, 4, d_model)

# Funcao cross_attention
def cross_attention(encoder_out, dec_state):

    # Matrizes de pesos aleatórias
    W_q = np.random.randn(d_model, d_k)
    W_k = np.random.randn(d_model, d_k)
    W_v = np.random.randn(d_model, d_v)
    
    # Projetando Q (do Decoder), K e V (do Encoder) usando o operador @
    Q = dec_state @ W_q
    K = encoder_out @ W_k
    V = encoder_out @ W_v

    # Calculando os scores de atenção
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
    
    # Aplicando softmax
    e_scores          = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = e_scores / np.sum(e_scores, axis=-1, keepdims=True)

    # Calculando a saída da atenção cruzada
    output = attention_weights @ V  # (1, 4, 64)
    return output

# Chamando a funcao cross_attention
cross_attention_output = cross_attention(encoder_output, decoder_state)
print("Saída da Atenção Cruzada (shape):")
print(cross_attention_output.shape)