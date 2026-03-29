from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
phrase    = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
tokens    = tokenizer.tokenize(phrase)

print(f"Frase original: \n{phrase}\n")
print(f"WordPiece: \n{tokens}\n")
