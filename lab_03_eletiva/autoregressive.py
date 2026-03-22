import numpy as np

# Simulando o Loop de Inferencia Auto-Regressivo

def generate_next_token(current_sequence, encoder_out):
    # Vetor de probabilidades fictício para o próximo token (supondo um vocabulário de 10000 tokens)
    vocab_size        = 10000
    next_token_probs  = np.random.rand(vocab_size)

    return next_token_probs

# Simulando a sequência de entrada (pode ser um token de início ou uma sequência inicial)
encoder_out_mock  = None
current_sequence  = ["<start>"]
EOS_TOKEN_ID      = 9999
max_len           = 10

print(f"Inicio: {current_sequence}")

while len(current_sequence) < max_len:
    probs = generate_next_token(current_sequence, encoder_out_mock)
    next_token_id = np.argmax(probs)

    if next_token_id == EOS_TOKEN_ID:
        current_sequence.append("<EOS>")
        break

    current_sequence.append(str(next_token_id))

print(f"Frase Final Gerada")
print(" ".join(current_sequence))
