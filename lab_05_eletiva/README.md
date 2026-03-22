# Laboratório Técnico 05: Treinamento Fim-a-Fim do Transformer

Implementação da rotina de treinamento completa em PyTorch para a arquitetura Transformer. 

## Funcionalidades
* Carregamento e tokenização de dataset real (`bentrevett/multi30k`) via Hugging Face.
* Refatoração das classes (Encoder, Decoder, Attention) de NumPy para PyTorch (`nn.Module`).
* Training Loop com mini-batches, otimizador Adam e CrossEntropyLoss.
* Teste de Overfitting com inferência auto-regressiva.

## Nota de Integridade
Partes geradas/complementadas com IA, revisadas por Antonio Gleyser Santos Mendes Brandão.
A IA foi utilizada para auxiliar na sintaxe de migração das matrizes NumPy para módulos PyTorch. Toda a topologia, o fluxo bidirecional de tensores e o laço de épocas (Forward/Backward/Step) foram codificados e validados manualmente.

## Fontes Usadas
https://huggingface.co/docs/datasets/en/loading
https://huggingface.co/google-bert/bert-base-multilingual-cased
https://nlp.seas.harvard.edu/annotated-transformer/#model-architecture
