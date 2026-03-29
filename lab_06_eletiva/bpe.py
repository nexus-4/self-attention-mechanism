# Motor de frequencia

import re

vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3,
}

def get_stats(vocab):
    # Recebe dicionario
    pairs = {}

    # varre cada palavra e olha os simbolos dois a dois
    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])

            # Somando a frequencia da palavra para o par
            if pair in pairs:
                pairs[pair] += freq
            else:
                pairs[pair] = freq
    
    return pairs

# Validacao da funcao get_stats
estatisticas = get_stats(vocab)

print("todas as frequencias: ")
print(estatisticas)

# buscando o par ('e', 's') para validacao
freq_es = estatisticas[('e', 's')]
print("frequencia do par ('e', 's'): ", freq_es) # retorno esperado: 9 

# ---------

def merge_vocab(pair, vocab_in):
    vocab_out = {}
    bigram    = re.escape(' '.join(pair))
    p         = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for word in vocab_in:
        word_out            = p.sub(''.join(pair), word)
        vocab_out[word_out] = vocab_in[word]
    
    return vocab_out

# LOOP DE TREINAMENTO (K=5)

print("INICIANDO TREINAMENTO DO TOKENIZADOR BPE (K=5)")

num_merges = 5

for i in range(num_merges):
    # Pega as estatísticas atuais
    estatisticas_atuais = get_stats(vocab)
    
    # Prevenção: se não houver mais pares para juntar, encerra o loop
    if not estatisticas_atuais:
        break
        
    # Encontra o par com a maior frequência
    melhor_par = max(estatisticas_atuais, key=estatisticas_atuais.get)
    
    print(f"\n[Iteração {i+1}]")
    print(f"Par mais frequente fundido: {melhor_par} (Frequência: {estatisticas_atuais[melhor_par]})")
    
    # Atualiza o vocabulário com a fusão
    vocab = merge_vocab(melhor_par, vocab)
    
    # Imprime o estado do dicionário
    print("Estado do vocabulário:")
    for palavra, freq in vocab.items():
        print(f"  {palavra}: {freq}")
        
print("Concluido")
