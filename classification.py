from tqdm import tqdm
from dotenv import load_dotenv
import os
import openai
import json

tqdm.pandas()
load_dotenv()

uni_api_key = os.getenv('UNI_API_KEY')

client = openai.OpenAI(
    api_key=uni_api_key,
    base_url="https://llms-inference.innkube.fim.uni-passau.de" )



with open('dict.json', 'r') as file:
    relations = json.load(file)
    
    
    
def classify_relation(prompt: str, model='llama3.1'):
    
    instruction = """
    You are a helpful assistant. Determine whether the given relation is a static or a dynamic relation.
    A static relation is a relation that doesn't change over time, means that the associated facts remain the same over time.
    While a dynamic relation changes over time, means that the associated facts change over time and have to be updated. 
    example: Cristiano Ronaldo is Born in Funchal, Portugal is a static fact, while Cristiano Ronaldo plays for Al-Nassr is a dynamic fact.
    Answer with "static" or "dynamic" only, no explanation is required.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instruction
            },

            {
                "role": "user",
                "content": "relation: " + prompt
            }]
        )
    return response.choices[0].message.content


if __name__=='__main__':
    
    relations_dict = {}
    
    for rel in tqdm(relations.values(), desc="Classifying relations"):
        relations_dict[rel] = classify_relation(rel)

    with open("relations.dict", "w") as f:
        json.dump(relations_dict, f, indent=4)
