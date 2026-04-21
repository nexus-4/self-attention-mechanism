# Laboratório 08: Alinhamento Humano com DPO (The HHH Dataset)

## Tema Escolhido
**Project Hail Mary (Astrofísica, Astrofagem e Sobrevivência)**

## Arquitetura e Engenharia
1. **Dataset de Preferências:** Gerado via script (`dpo_dataset_generator.py`) com Google GenAI (gemini-3.1-pro-preview). Contém arrays com `prompt`, `chosen` (seguro) e `rejected` (inadequado).

2. **Treinamento DPO:** Executado no Google Colab (`dpo_trainer.ipynb`) via GPU T4, utilizando otimização de memória `paged_adamw_32bit`.

3. **Resolução de Hardware:** Implementação de script para cast iterativo de tensores (`_amp_foreach_non_finite_check_and_unscale_cuda`), forçando a conversão de parâmetros `BFloat16` para `Float32`, viabilizando o uso da arquitetura Turing.

## A Engenharia do Hiperparâmetro Beta no DPO
A Otimização Direta de Preferência (DPO) ajusta o modelo usando a Divergência Kullback-Leibler (KL). O hiperparâmetro $\beta$ (configurado em 0.1 neste lab) atua como um coeficiente de penalidade. Ele age como um "imposto" regulatório: impede que a nova política do modelo se desvie excessivamente da política do modelo base congelado. Se $\beta$ for alto demais, o modelo ignora o alinhamento; se for baixo demais, ocorre "Mode Collapse" (destruição da fluência para agradar regras). O valor de 0.1 garante o balanço ideal entre segurança HHH e qualidade de linguagem.

## Declaração de Integridade
Partes geradas/complementadas com IA, revisadas por Antonio Gleyser Santos Mendes Brandao.
(O uso da IA na parte de debugging na resolução de incompatibilidade de tensores FP16/BF16 durante o setup do `DPOTrainer`).