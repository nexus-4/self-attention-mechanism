import numpy as np
import pandas as pd

# Frase inicial
frase       = "Fear is the mind killer"
palavras    = frase.split()
vocabulario = list(set(palavras))

# Criando um dicionário de palavras para índices
word_to_index = {palavra: idx for idx, palavra in enumerate(vocabulario)}

# Criando um dicionário de índices para palavras
index_to_word = {idx: palavra for palavra, idx in word_to_index.items()}

# Convertendo a frase em uma sequência de índices
sequencia_indices = [word_to_index[palavra] for palavra in palavras]

print("Frase:", frase)
print("Palavras:", palavras)
print("Vocabulário:", vocabulario)
print("Dicionário de palavras para índices:", word_to_index)
print("Dicionário de índices para palavras:", index_to_word)
print("Sequência de índices:", sequencia_indices)

# Para o DataFrame
df_vocab = pd.DataFrame(list(word_to_index.items()), columns=['Palavra', 'Índice'])
print(df_vocab)

# Criando a tabela de embeddings
d_model     = 64
embeddings  = np.random.randn(len(vocabulario), d_model)
print("Tabela de embeddings (shape):", embeddings.shape)

# Montar o tensor de Entrada X
X_2d = embeddings[sequencia_indices]

X    = np.expand_dims(X_2d, axis=0)

print("Tensor de Entrada X (shape):", X.shape)
