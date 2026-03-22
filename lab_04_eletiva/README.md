# Laboratório Técnico 04: O Transformer Completo "From Scratch"

Este repositório contém a implementação final da arquitetura completa (Encoder-Decoder) do Transformer. O projeto integra os blocos construídos nos laboratórios anteriores e realiza um teste de inferência fim-a-fim, acoplando um laço auto-regressivo para processar a toy sequence "Thinking Machines".

## Estrutura do Projeto
* `modules.py`: Blocos base refatorados (Scaled Dot-Product Attention universal, Feed-Forward Network com expansão para 2048, e Add & Norm).

* `encoder.py`: Classe `EncoderBlock` implementando o fluxo bidirecional de contexto.

* `decoder.py`: Classe `DecoderBlock` implementando o Masked Self-Attention (com look-ahead mask) e a ponte de Cross-Attention.

* `transformer.py`: Topologia central unindo Encoder e Decoder, com projeção Linear e Softmax final para o tamanho do vocabulário.

* `main.py`: Script de prova final rodando o laço auto-regressivo iterativo até a geração do token `<EOS>`.

## Como Executar
Certifique-se de estar no seu ambiente virtual com as dependências instaladas ( NumPy) e execute o orquestrador:

bash
python main.py


O console exibirá o passo a passo da geração de tokens do Decoder baseada nas probabilidades do Softmax.

## Nota de Crédito e Integridade
Partes geradas/complementadas com IA, revisadas por Antonio Gleyser. 

**Detalhamento do uso de IA:** A inteligência artificial (Gemini) foi utilizada exclusivamente para brainstorming da arquitetura de arquivos, geração de esqueletos básicos (templates) e depuração de sintaxe da biblioteca NumPy (como o ajuste da transposição de tensores 3D com `np.swapaxes`). O fluxo bidirecional, a lógica matemática de montagem dos blocos Encoder/Decoder e o laço auto-regressivo de inferência foram codificados, compreendidos e validados manualmente.