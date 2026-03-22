import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import tokenize_sequences, prepare_dataset, tokenizer
from model import Transformer

print("="*50)
print("PREPARANDO DADOS E MODELO")
print("="*50)

dataset_raw             = prepare_dataset()
enc_inputs, dec_inputs  = tokenize_sequences(dataset_raw)
vocab_size              = tokenizer.vocab_size
pad_token_id            = tokenizer.pad_token_id 

d_model = 128
model   = Transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, d_model=d_model, d_ff=512)

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

print("Iniciando Treinamento com Mini-Batches...")
epochs      = 8
batch_size  = 32

for epoch in range(epochs):
    model.train()
    
    total_loss = 0
    
    for i in range(0, len(enc_inputs), batch_size):
        src_batch = enc_inputs[i : i + batch_size]
        dec_batch = dec_inputs[i : i + batch_size]
        
        # Aplicando o Teacher Forcing no batch atual
        tgt_input = dec_batch[:, :-1]
        tgt_expected = dec_batch[:, 1:]
        
        seq_len = tgt_input.size(1)
        mask = create_causal_mask(seq_len)
        
        optimizer.zero_grad()
        
        logits = model(src_batch, tgt_input, target_mask=mask)
        
        logits_flat = logits.reshape(-1, vocab_size)
        tgt_expected_flat = tgt_expected.reshape(-1)
        
        loss = criterion(logits_flat, tgt_expected_flat)
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        
    media_loss = total_loss / (len(enc_inputs) / batch_size)
    print(f"Época [{epoch+1}/{epochs}] | Loss Média: {media_loss:.4f}")


# ==========================================
# TAREFA 4: A PROVA DE FOGO (INFERÊNCIA AUTO-REGRESSIVA)
# ==========================================
print("\n" + "="*50)
print("TAREFA 4: TESTE DE OVERFITTING (INFERÊNCIA)")
print("="*50)

# 1. Escolhendo a frase de teste (Amostra 3)
idx_teste = 2 # Índice 2 é a terceira frase
frase_ingles = dataset_raw[idx_teste]['en']
frase_alemao_real = dataset_raw[idx_teste]['de']

print(f"Frase Original (EN): {frase_ingles}")
print(f"Tradução Esperada (DE): {frase_alemao_real}")

# Colocando o modelo em modo de avaliação (desliga dropout/atualização de pesos)
model.eval()

# 2. Preparando a entrada do Encoder
enc_input_teste = tokenizer(
    frase_ingles, 
    return_tensors='pt',
    max_length=32,
    truncation=True,
    padding='max_length'
)['input_ids']

with torch.no_grad(): # Sem cálculo de gradientes para economizar memória
    src_emb = model.src_embedding(enc_input_teste)
    Z = model.encoder(src_emb) # A Matriz de Memória Z

# 3. Iniciando o Decoder
# No modelo BERT, [CLS] é o <START> e [SEP] é o <EOS>
start_token = tokenizer.cls_token_id
eos_token = tokenizer.sep_token_id

decoder_sequence = [start_token]
max_len_inferencia = 32

print("\nGerando tradução token por token...")

for step in range(max_len_inferencia):
    # Transforma a lista numa matriz PyTorch [1, seq_len]
    tgt_tensor = torch.tensor([decoder_sequence])
    
    # Cria a máscara para o tamanho atual
    seq_len_atual = tgt_tensor.size(1)
    mask_teste = create_causal_mask(seq_len_atual)
    
    with torch.no_grad():
        tgt_emb = model.tgt_embedding(tgt_tensor)
        # O pulo do gato: passando o target gerado e a memória Z
        dec_out = model.decoder(tgt_emb, Z, target_mask=mask_teste)
        logits = model.fc_out(dec_out)
    
    # Pega as probabilidades apenas da ÚLTIMA palavra gerada
    next_token_logits = logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    
    decoder_sequence.append(next_token_id)
    
    # Se o modelo prever o fim da frase, interrompe o laço
    if next_token_id == eos_token:
        break

# 4. Decodificando os IDs de volta para texto legível
# skip_special_tokens=True limpa os [CLS], [SEP] e [PAD] da string final
traducao_gerada = tokenizer.decode(decoder_sequence, skip_special_tokens=True)

print(f"\n-> Tradução Gerada pelo Modelo: {traducao_gerada}")
print("="*50)