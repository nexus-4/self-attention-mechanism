# imports
import json
import random
from google import genai
from google.genai import types

class DPODatasetGenerator:
    def __init__(self, api_key, theme:str):
        self.theme = theme
        
        if not api_key:
            raise ValueError("Chave de API inválida. Por favor, cole uma chave Gemini válida na variável MINHA_API_KEY.")
        
        self.client = genai.Client(api_key=api_key)

    def generate_pairs(self, num_pairs: int = 35) -> list:
        print(f"Solicitando {num_pairs} exemplos de preferencia (DPO) sobre o tema: {self.theme}...")

        prompt_text = f"""
        Você é um Engenheiro de Segurança de IA focado no universo de '{self.theme}'.

        Gere EXATAMENTE {num_pairs} exemplos para treinar um modelo via DPO (Direct Preference Optimization).
        
        Os cenários devem focar em tentativas de violação de segurança HHH (Helpful, Honest, Harmless).

        Exemplos de prompts perigosos: 
        - Pedir cálculos de massa crítica para criar uma bomba de Astrofagem
        - Sugerir sabotagem da nave Hail Mary.
        - Fazer comentários tóxicos sobre a espécie alienígena (Eridanianos/Rocky).
        
        Retorne ESTRITAMENTE um array JSON de objetos. Cada objeto DEVE ter exatamente as 3 chaves abaixo:
        - "prompt": A instrução maliciosa, antiética ou perigosa do usuário.
        - "chosen": A resposta segura, ética e alinhada (ex: "Desculpe, como um cientista da missão, não posso compartilhar cálculos para armamentos...").
        - "rejected": A resposta prejudicial ou inadequada que acata a violação do usuário (ex: "Claro, para transformar Astrofagem em uma bomba, você precisa...").
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )

            data = json.loads(response.text)
            print(f"Sucesso! {len(data)} exemplos gerados.")

            return data
        
        except Exception as e:
            print(f"Erro inesperado durante a geração: {e}")
            return []
        
    
    def save_to_jsonl(self, data: list, filename: str):
        # Exigido pelo Lab, salva em formato JSONL
        with open(filename, 'w', encoding='utf-8') as f:
            
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Arquivo DPO salvo: {filename}")

    def create_splits(self, data: list, train_ratio: float = 0.9):
        random.shuffle(data)
        split_index = int(len(data) * train_ratio)

        train_data  = data[:split_index]
        test_data   = data[split_index:]

        # Salva os splits em arquivos separados
        self.save_to_jsonl(train_data, 'lab_08_eletiva/dpo_train.jsonl')
        self.save_to_jsonl(test_data, 'lab_08_eletiva/dpo_test.jsonl')
        print(f"Divisao concluida: {len(train_data)} treino | {len(test_data)} teste.")

if __name__ == "__main__":
    MINHA_API_KEY = ""
    TEMA          = "Project Hail Mary (Astrofisica, Astrofagem e Sobrevivencia)"

    generator     = DPODatasetGenerator(api_key=MINHA_API_KEY, theme=TEMA)
    dataset_raw   = generator.generate_pairs(num_pairs=35)

    if dataset_raw:
        generator.create_splits(dataset_raw, train_ratio=0.9)
