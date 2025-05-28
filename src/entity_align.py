import re
import pandas as pd
from fuzzywuzzy import fuzz
from difflib import get_close_matches
import requests





def extract_answer(text):
    pattern = r"Answer:\s*(.+?)(?=\s*(?:, Confidence|\nConfidence|$))"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None

def extract_explanation(text):
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def fuzzy_match(name1, name2, threshold=80):
    similarity_score = fuzz.ratio(name1, name2)
    return similarity_score >= threshold


def find_closest_entity(label, all_labels):
    match = get_close_matches(label, all_labels, n=1, cutoff=0.7)
    return match[0] if match else None



def search_wikidata_entity(name):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json",
        "limit": 10,
        #"strictlanguage": False
    }
    response = requests.get(url, params=params)
    data = response.json()
    try:
        return data['search'][0]['label'] if data['search'] else None
    except:
        return None


# def wikidata_linker():
#     nlp = spacy.load("en_core_web_md")
#     nlp.add_pipe("entityLinker", last=True)
#     doc = nlp("Mukesh Ambani")
#     all_linked_entities = doc._.linkedEntities
#     return all_linked_entities[0]


if __name__=='__main__':
    
    
    df = pd.read_csv('bank_governors_results.csv')
    df['answer']=df['prediction'].apply(extract_answer)
    df['expalanation'] = df['prediction'].apply(extract_explanation)
    df['confidence'] = df['prediction'].str.extract(r'(?i)\b(high|medium|low)\b')
    df['sim']=df.apply(lambda row: fuzzy_match(row['governorLabel'], row['answer']), axis=1)
    all_names = df['governorLabel'].tolist()

    df.loc[~df['sim'], 'closest_match'] = df.loc[~df['sim'], 'answer'].apply(
        lambda x: find_closest_entity(x, all_names)
    )
    df.loc[df['closest_match'].isna(), 'wiki_label']=df.loc[df['closest_match'].isna(), 'answer'].apply(
    lambda x: search_wikidata_entity(x)
    )
    
    print(df)