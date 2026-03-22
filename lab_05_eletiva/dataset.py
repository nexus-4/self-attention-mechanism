from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# Instanciando o tokenizador usando o modelo "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def prepare_dataset():
    # Carregando o dataset sugerido bentrevett/multi30k com apenas 1000 amostras para treinamento
    dataset = load_dataset("bentrevett/multi30k", split="train[:1000]")

    return dataset

dataset = prepare_dataset()

for i in range(5):
    print(f"Amostra {i+1}:")
    print(f"Frase em Inglês: {dataset[i]['en']}")
    print(f"Frase em Alemão: {dataset[i]['de']}")
    print("-" * 50)

def tokenize_sequences(dataset):
    max_length = 32

    english_sentences = [item['en'] for item in dataset]
    german_sentences  = [item['de'] for item in dataset]

    enc_encoded = tokenizer(
        english_sentences,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    dec_encoded = tokenizer(
        german_sentences,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


    return enc_encoded['input_ids'], dec_encoded['input_ids']

# Testando a função
enc_inputs, dec_inputs = tokenize_sequences(dataset)
print(f"Formato do tensor do Encoder (Inglês): {enc_inputs.shape}")
print(f"Formato do tensor do Decoder (Alemão): {dec_inputs.shape}")