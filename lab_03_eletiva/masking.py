import numpy as np

# Implementando a mascara causal (look-Ahead Mask)

def create_causal_mask(seq_len):
    # Criando uma matriz quadrada de tamanho (seq_len, seq_len), onde a diagonal e o tringualo inferior sao zeros e o tringulo superior sao -infinito

    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf

    return mask

# Definindo os parametros 
seq_len = 5
d_k     = 64

# Matrizes ficticias de Q e K
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)

# Produto escalar
scores = np.dot(Q, K.T) / np.sqrt(d_k)

# Somando o resultado anterior com a mascara gerada pela funcao create_causal_mask

mask           = create_causal_mask(seq_len)
masked_scores  = scores + mask

# Softmax para obter as probabilidades de atenção
attention_weights = np.exp(masked_scores) / np.sum(np.exp(masked_scores), axis=-1, keepdims=True)

print("Scores antes da mascara:")
print(scores)

print("\nMascara Causal:")
print(mask)

print("\nScores depois da mascara:")
print(masked_scores)

print("\nProbabilidades de Atenção:")
print(attention_weights)
