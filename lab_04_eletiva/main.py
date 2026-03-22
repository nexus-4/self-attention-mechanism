import numpy as np
from transformer import Transformer

# Máscara Causal
def create_causal_mask(seq_len):
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask

print("INICIANDO INFERÊNCIA DO TRANSFORMER COMPLETO")
print("="*50)

vocab_size = 100
d_model    = 512
model      = Transformer(vocab_size=vocab_size, d_model=d_model, d_ff=2048)

# 2. Tensor de entrada simulando "Thinking Machines" (2 tokens)
print("-> Codificando a entrada 'Thinking Machines'...")
encoder_input = np.random.randn(1, 2, d_model)

# 3. Laço Auto-Regressivo
START_TOKEN = 0
EOS_TOKEN   = 99
max_len     = 10 # Trava de segurança para não rodar infinitamente

decoder_sequence = [START_TOKEN]
print(f"-> Iniciando Decoder com <START> (ID: {START_TOKEN})")

while len(decoder_sequence) < max_len:
    seq_len = len(decoder_sequence)
    
    # Criamos um tensor Y fictício que cresce a cada iteração
    y_mock = np.random.randn(1, seq_len, d_model)
    
    # Criamos a máscara causal para o tamanho atual da sequência
    mask = create_causal_mask(seq_len)
    
    # Forward Pass na arquitetura completa!
    probabilities = model.forward(encoder_input, y_mock, mask)
    
    # Pegamos as probabilidades apenas da ÚLTIMA palavra gerada
    next_token_probs = probabilities[0, -1, :]
    
    # Selecionamos a palavra com maior probabilidade
    next_token_id = np.argmax(next_token_probs)
    
    if next_token_id == EOS_TOKEN:
        decoder_sequence.append("<EOS>")
        print(f"   Gerou: <EOS> (Fim da iteração)")
        break
        
    decoder_sequence.append(int(next_token_id))
    print(f"   Gerou Token ID: {next_token_id} | Sequência Atual: {decoder_sequence}")

print("\n" + "="*50)
print("INFERÊNCIA CONCLUÍDA COM SUCESSO!")
print("="*50)