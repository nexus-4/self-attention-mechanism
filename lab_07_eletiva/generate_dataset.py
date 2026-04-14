import json
import random
from google import genai
from google.genai import types

class SyntheticDatasetGenerator:
    def __init__(self, api_key: str, theme: str):
        self.theme = theme
        self.client = genai.Client(api_key=api_key)

    def generate_pairs(self, num_pairs: int = 50) -> list:
        print(f"Solicitando {num_pairs} pares ao Gemini sobre o tema: {self.theme}...")
        
        prompt_text = f"""
        Você é um especialista no universo do livro '{self.theme}' de Andy Weir.
        Gere exatamente {num_pairs} exemplos de interações (Q&A) entre um usuário 
        e um assistente de IA focado em astrofísica, biologia de sobrevivência, 
        e a ciência de Astrofagem (Astrophage).
        
        As respostas devem ter um tom pragmático, científico e levemente 
        sarcástico, lembrando o personagem Ryland Grace ou explicando 
        conceitos com a lógica do personagem alienígena Rocky ("Amaze!").
        
        Retorne um array de objetos, onde cada objeto tem estritamente duas chaves:
        "prompt" (a pergunta do usuário) e "response" (a resposta do assistente).
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                ),
            )
            
            data = json.loads(response.text)
            print(f"Sucesso! {len(data)} pares gerados.")
            return data
            
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar o JSON: {e}")
            print("Resposta bruta do modelo (para debug):", response.text)

            return []
        except Exception as e:
            print(f"Erro inesperado: {e}")

            return []

    def save_to_jsonl(self, data: list, filename: str):

        with open(filename, 'w', encoding='utf-8') as f:

            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Arquivo salvo: {filename}")

    def create_splits(self, data: list, train_ratio: float = 0.9):
        random.shuffle(data)
        split_index = int(len(data) * train_ratio)
        
        train_data  = data[:split_index]
        test_data   = data[split_index:]
        
        self.save_to_jsonl(train_data, 'train_dataset.jsonl')
        self.save_to_jsonl(test_data, 'test_dataset.jsonl')
        print(f"Divisão concluída: {len(train_data)} treino | {len(test_data)} teste.")

if __name__ == "__main__":
    MINHA_API_KEY = "AIzaSyA7_EhmrAGeHpYF7okX0xG8lZFaVqGSm5M" 
    
    TEMA_ESCOLHIDO = "Project Hail Mary (Astrofísica e Sobrevivência Espacial)" 
    
    # Instancia a classe
    generator = SyntheticDatasetGenerator(api_key=MINHA_API_KEY, theme=TEMA_ESCOLHIDO)
    
    # O Laboratório exige no mínimo 50 pares 
    dataset_raw = generator.generate_pairs(num_pairs=55) 
    
    if dataset_raw:
        generator.create_splits(dataset_raw, train_ratio=0.9)