# Laboratório 6: Construindo um Tokenizador BPE e Explorando o WordPiece

## Análise do WordPiece (Tarefa 3)
 - Ao analisar os tokens gerados pelo `bert-base-multilingual-cased`, podemos notar o uso frequente do prefixo `##` (ex: `##mente`). No algoritmo WordPiece, o sinal de cerquilha dupla indica que aquele token é uma sub-palavra e deve ser anexado diretamente ao token anterior, sem espaço em branco na hora de reconstruir a frase original.

 - O uso de algoritmos de sub-palavras (como BPE e WordPiece) impede o travamento do modelo diante de vocabulário desconhecido (o problema de *Out-Of-Vocabulary* - OOV). Em vez de descartar uma palavra complexa ou inédita (como "inconstitucionalmente") substituindo-a por um token de erro, o modelo consegue fatiá-la em radicais, prefixos e sufixos previamente conhecidos (ex: "in", "##cons", "##tit", "##uc", "##ional", "##mente"). Isso permite que a rede neural preserve a semântica morfológica e continue processando o texto sem falhas dimensionais.

## Declaração de Uso de IA
 - Conforme as diretrizes da disciplina, declaro que a inteligência artificial generativa foi utilizada para auxiliar na construção do loop principal de treinamento do tokenizador. O código gerado foi validado, revisado e testado manualmente para garantir o cumprimento total dos requisitos.

## Fontes
- https://huggingface.co/google-bert/bert-base-multilingual-cased
- https://medium.com/@adarsh-ai/build-a-byte-pair-encoding-bpe-tokenizer-from-scratch-in-python-0dc32c6410f7