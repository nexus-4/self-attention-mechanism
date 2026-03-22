# Laboratório 3: Implementando o Decoder do Transformer

Este repositório contém os blocos matemáticos centrais do Decoder, construídos do zero com Python e NumPy, garantindo que o modelo gere texto de forma fluente e sem "olhar para o futuro"[cite: 100].

## Estrutura do Projeto
1. **Máscara Causal**: Implementação da álgebra linear para o Look-Ahead Masking[cite: 103].
2. **Cross-Attention**: A ponte de integração estrutural projetando Queries do Decoder e Keys/Values do Encoder.
3. **Loop Auto-Regressivo**: Simulação do laço de repetição condicionado na produção anterior até a geração do token `<EOS>`.

## Como executar
Para rodar todas as provas reais e visualizar as saídas dos tensores sequencialmente, execute o arquivo orquestrador:
\`\`\`bash
python3 main.py
\`\`\`

## Nota de Crédito (Uso de IA)
Em conformidade com as regras do laboratório, o uso de IA Generativa (Gemini) foi restrito exclusivamente a tirar dúvidas de sintaxe do `numpy` (broadcasting, manipulação de eixos) e brainstorming estrutural. Nenhuma classe **pronta** gerada por máquina foi submetida. Toda a lógica matemática e arquitetura dos arquivos foi desenvolvida manualmente.