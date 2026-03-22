# Laboratório 02: Construindo o Transformer Encoder "From Scratch"

Este repositório contém a implementação da passagem direta (Forward Pass) de um bloco Encoder do Transformer, baseado no artigo "Attention Is All You Need", utilizando apenas Python puro, NumPy e Pandas.

## Pré-requisitos
Certifique-se de ter o Python 3.x instalado. Instale as dependências executando:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Como executar
Para rodar a validação e visualizar a passagem do tensor pelo Encoder, execute o arquivo principal:
\`\`\`bash
python3 main.py
\`\`\`
O console exibirá o processamento do vocabulário, o formato do tensor de entrada $X$ e o formato do tensor de saída $Z$ após passar por $N=6$ camadas.

## Nota de Crédito (Uso de IA)
Conforme as diretrizes do laboratório, o uso de IA Generativa é permitido apenas para tirar dúvidas de sintaxe do \`numpy\` e brainstorming. Utilizei o Gemini exclusivamente como auxílio para a estruturação lógica do código, cálculos de dimensões (broadcasting) e depuração matemática. Nenhuma classe pronta gerada pela máquina foi submetida. Todo o código final foi implementado e arquitetado manualmente.