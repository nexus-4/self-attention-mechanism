import numpy as np
from attention import ScaledDotProductAttention

def run_test():
    # Inicializando a camada de atenção
    attention_layer = ScaledDotProductAttention()

    # Exemplo simples com 3 tokens("The One Ring") e dimensão de embedding = 4
    np.random.seed(67)
    Q = np.random.rand(3, 4)
    K = np.random.rand(3, 4)
    V = np.random.rand(3, 4)

    print("Matrizes de Entrada")
    print(f"Shape Q: {Q.shape}")
    print(f"Shape K: {K.shape}")
    print(f"Shape V: {V.shape}\n")

    # Calculando a atenção
    output, weights = attention_layer.forward(Q, K, V)

    print("Saídas")
    print(f"Shape do Output: {output.shape}")
    print("Matriz de Output:")
    print(np.round(output, 4))
    
    print(f"\nShape dos Pesos de Atenção (Softmax): {weights.shape}")
    print("Pesos (Cada linha deve somar 1):")
    print(np.round(weights, 4))
    
    # Validação simples da soma do softmax na primeira linha
    print(f"\nSoma da primeira linha do Softmax: {np.sum(weights[0]):.2f}")

if __name__ == "__main__":
    run_test()