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


# input_file = "humans_wikidata/humans_wikidata/type2relation2type_ttv.txt"

# unique_relations = set()

# with open(input_file, "r", encoding="utf-8") as f:
#     for line in f:
#         parts = line.strip().split("\t")
#         if len(parts) >= 2:
#             relation = parts[1].strip()
#             if "/relation/" in relation:
#                 rel_id = relation.split("/")[-1]
#                 unique_relations.add(rel_id)

# sorted_relations = sorted(unique_relations)

# entities_dict, relations_dict = entity2id_codex()
# human_wikidata_relations = []

# for rel in sorted_relations:
#     relation = relations_dict.get(rel, None)
#     if relation:
#         human_wikidata_relations.append(relation)