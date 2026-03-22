from data_preparation import X 
from layers import EncoderLayer, TransformerEncoder

print(f"Formato do Tensor de Entrada (X): {X.shape}") 

# Instanciando o modelo com 6 camadas e d_model = 64
encoder = TransformerEncoder(num_layers=6, d_model=64)

# Realizando a passagem direta (Forward Pass)
Z = encoder.forward(X)

print(f"Formato do Tensor de Saída (Vetor Z): {Z.shape}")
