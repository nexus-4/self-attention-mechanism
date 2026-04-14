# Laboratório 07: Especialização de LLMs com LORA e QLORA

## Tema do Dataset Sintético
**Project Hail Mary (Astrofísica e Sobrevivência Espacial)**
O dataset foi gerado utilizando a API do Gemini 2.5 Flash, simulando interações (Q&A) de um assistente especializado na ciência e nas dinâmicas de sobrevivência abordadas no livro *Project Hail Mary* de Andy Weir, incluindo o comportamento da Astrofagem e o tom pragmático/sarcástico do protagonista Ryland Grace e do alienígena Rocky ("Amaze!").

## Arquitetura do Pipeline
1. **Geração de Dados:** Script Python Orientado a Objetos (`dataset_generator.py`) utilizando `google-genai`.
2. **Treinamento SFT:** Jupyter Notebook executado no Google Colab (`lora_trainer.ipynb`) utilizando GPU T4.
3. **Quantização:** Modelo base carregado em 4-bits (`nf4`) com `bitsandbytes`.
4. **PEFT/LORA:** Matrizes de decomposição configuradas com Rank `64`, Alpha `16` e Dropout `0.1`.
5. **Otimização:** Uso do otimizador `paged_adamw_32bit` com scheduler de aprendizado tipo `cosine`.

## Declaração de Uso de IA
Partes geradas/complementadas com IA, revisadas por Antonio Gleyser Santos Mendes Brandao.
(O uso da IA no suporte na resolução de conflitos de versão das bibliotecas `trl` e `pyarrow` durante o setup do SFTTrainer no Colab).