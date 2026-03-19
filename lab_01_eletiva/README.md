# Lab P1-01: Implementação de Self-Attention

Este repositório contém a implementação do mecanismo de *Scaled Dot-Product Attention*, conforme descrito no artigo *"Attention Is All You Need"*. O código foi desenvolvido inteiramente usando a biblioteca NumPy, sem depender de frameworks de Deep Learning (como Keras ou PyTorch).

## 🗂 Estrutura do Repositório
* `attention.py`: Contém a classe `ScaledDotProductAttention` e a implementação do Softmax.
* `test_attention.py`: Script para execução de testes unitários básicos e validação numérica.
* `README.md`: Documentação atual do projeto.

## 🚀 Como rodar o código

1. Certifique-se de ter o Python instalado.
2. Instale o NumPy caso não possua:
   ```bash
   pip install numpy
   ```
3. Execute o script de testes para validar a implementação:

  ```bash
   python test_attention.py
   ```



## 🧠 A Normalização (Scaling Factor)

No cálculo do Attention, o produto escalar entre Q e Kt é dividido pela raiz quadrada da dimensão das chaves, representada por √dk.

Isso é necessário para evitar que os valores do produto escalar fiquem excessivamente grandes quando trabalhamos com altas dimensões. Valores muito altos empurrariam a função Softmax para regiões extremas onde os gradientes são muito pequenos (problema de *vanishing gradients*), prejudicando o aprendizado da rede. A divisão atua como um estabilizador numérico.

## 📊 Exemplo de Input e Output Esperado

Ao rodar o script de teste, a validação ocorre com os seguintes dados:

* **Input (Q, K, V):** Matrizes aleatórias de formato `(3, 4)`, representando 3 tokens com dimensão de embedding igual a 4.
* **Output (Matriz de Atenção Resultante):** Formato `(3, 4)`, representando a nova projeção ponderada dos tokens.
* **Pesos de Atenção (Softmax):** Uma matriz intermediária de formato `(3, 3)`. A soma dos valores em cada linha dessa matriz é validada para garantir que resulte exatamente em `1.0`.

## 📚 Artigos como Referências

1. [Understanding Softmax with Numpy](https://medium.com/@amit25173/understanding-softmax-with-numpy-b7273d8ab205)
2. [Understanding the Attention Mechanism: A simple implementation using Python and Numpy](https://medium.com/@christoschr97/understanding-the-attention-mechanism-a-simple-implementation-using-python-and-numpy-3f1feae13fb7)
