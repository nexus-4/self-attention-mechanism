# Laboratório 09: Arquitetura RAG (HNSW, HyDE e Cross-Encoders)

## Arquitetura do Sistema RAG (Wallace Corp Knowledge Base - Blade Runner)
Este pipeline resolve o problema da dissonância de vocabulário entre queries humanas colloquiais e manuais corporativos altamente técnicos, adotando as seguintes etapas:
1. **Indexação FAISS (HNSW):** 20 manuais técnicos fictícios baseados na lore de Blade Runner 2049, vetorizados com `all-MiniLM-L6-v2`.
2. **HyDE (Hypothetical Document Embeddings):** Uso do `gemini-2.5-flash` para "alucinar" um texto técnico a partir da gíria de um policial (Query Transformation).
3. **Busca Rápida (Bi-Encoder):** Similaridade de Cosseno via `faiss` resgatando o Top-10.
4. **Re-Ranking (Cross-Encoder):** Refinamento da busca passando a query original e os candidatos pelo modelo `cross-encoder/ms-marco-MiniLM-L-6-v2`, filtrando para o Top-3 exato a ser consumido pelo LLM final.

## Tarefa Analítica: Consumo de RAM (HNSW vs KNN Exato)
A busca K-Nearest Neighbors (KNN Flat) exata apenas armazena os vetores numéricos na memória. O índice HNSW (Hierarchical Navigable Small World), por outro lado, exige significativamente **mais memória RAM**. Isso ocorre porque, além dos vetores originais, o HNSW armazena uma estrutura de grafos multi-camadas.
Os hiperparâmetros determinam este peso: o **M** define o número máximo de conexões bidirecionais que cada nó de vetor mantém no grafo, e o **ef_construction** aumenta o tamanho da fila de busca estendendo a varredura durante a criação. Isso aumenta violentamente o tamanho do índice armazenado na RAM em troca de realizar buscas em velocidade logarítmica extrema (milissegundos) em vez de velocidade linear lenta (varredura bruta do KNN).

## Declaração de Integridade
Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Antonio Gleyser.
(O uso de IA englobou a geração da base de dados fictícia técnica de Blade Runner 2049 e implementacao com FAISS).
