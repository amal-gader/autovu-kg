import re
import pandas as pd
from fuzzywuzzy import fuzz 


def entity2id_codex():
    with open('data/codex/entities.json', 'r') as file:
            entities = pd.read_json(file, orient='index')
    entities_dict = {index:row['label'] for index, row in entities.iterrows()}
    with open('data/codex/relations.json', 'r') as file:
            relations = pd.read_json(file, orient='index')
    relations_dict = {index:row['label'] for index, row in relations.iterrows()}
    return entities_dict, relations_dict

def extract_fields(text):
    pattern = (
        r"Answer:\s*(?P<answer>.*?)\s*"
        r"Confidence:\s*(?P<confidence>.*?)\s*"
        r"Reference:\s*(?P<reference>.*?)\s*$"
    )
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group("answer"), match.group("confidence"), match.group("reference")
    else:
        return None, None, None

def rowwise_eval(row1, row2, threshold=75):
    score = fuzz.ratio(str(row1).lower(), str(row2).lower())
    return True if score >= threshold else False



def jaccard_similarity(str1, str2, n=2):
    str1, str2 = str1.lower(), str2.lower()
    
    # Create n-grams for both strings
    ngrams1 = set([str1[i:i+n] for i in range(len(str1)-n+1)])
    ngrams2 = set([str2[i:i+n] for i in range(len(str2)-n+1)])
    
    # Calculate the intersection and union of the sets
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    
    return len(intersection) / len(union)


