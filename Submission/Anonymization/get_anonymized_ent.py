from spacy.tokens import Span
import spacy
import scispacy
from scispacy.linking import EntityLinker
import pandas as pd
import json
import tqdm
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import faiss
import pickle
# seed everything
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


nlp = spacy.load("en_core_sci_md")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")


def normalize_disease(entities, threshold=0.8):# Iterate over the list of entities and link them
    normalized_entities = []
    #for entity in entities:
    # tqdm
    for entity in tqdm.tqdm(entities, total=len(entities)):
        # Create a temporary Doc object for the entity text
        doc = nlp.make_doc(entity)
        
        # Manually create a Span for the entity, assuming it spans the whole doc
        span = Span(doc, 0, len(doc), label="ENTITY")
        doc.ents = [span]  # Set the ents attribute to the list containing the span
    
        # Use the EntityLinker to find potential matches
        linked_entities = linker(doc)

        # get the most confident concept name if score > threshold
        best_concept = None
        best_score = 0.0
        
        # Iterate through the linked entities
        for concept_id, score in span._.kb_ents:
            # Only consider concepts above the threshold
            if score > threshold and score > best_score:
                best_score = score
                best_concept = linker.kb.cui_to_entity[concept_id].canonical_name
        
        # Output the most confident concept name and its score
        if best_concept:
            normalized_entities.append(best_concept)
        else:
            normalized_entities.append(entity)

    return normalized_entities


def find_nearest_neighbors(matrix):
    # Convert the matrix to float32 (Faiss requires this format)
    matrix = matrix.astype(np.float32)
    
    # Create an index
    #sindex = faiss.IndexFlatL2(matrix.shape[1])  # L2 distance
    # cosine sim
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)  # Add vectors to the index
    
    # Query for the nearest neighbor for each vector
    distances, indices = index.search(matrix, 2)  # k=2 to exclude the point itself
    
    # Extract the indices of the nearest neighbors, ignoring self-comparison
    nearest_neighbor_indices = indices[:, 1]
    
    # Create the result matrix with the nearest neighbors
    nearest_neighbors = matrix[nearest_neighbor_indices]
    
    return nearest_neighbors, distances, indices, nearest_neighbor_indices

def find_neighbors_sampled(selected_X, sampled_X):
    # Convert matrices to float32 (Faiss requires this format)
    selected_X = selected_X.astype(np.float32)
    sampled_X = sampled_X.astype(np.float32)
    
    # Create an index for selected_X
    index = faiss.IndexFlatIP(sampled_X.shape[1])  # Cosine similarity
    
    # Add selected_X vectors to the index
    index.add(sampled_X)
    
    # Query the index with sampled_X to find the nearest neighbors in selected_X
    distances, indices = index.search(selected_X, 1)  # k=1 to find the single nearest neighbor
    
    
    return distances, indices


def main():
    data_path = '../NEextraction/entity_dict_negated.json'
    
    # extracting only PROBLEM
    # iter through rows
    doc_ent_list = []
    id_list  = []
    
    start_tag_regex = re.compile(r'<e\d+>')
    end_tag_regex = re.compile(r'</e\d+>')
    
    
    
    # read as lines of json
    json_list = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            json_list.append(json.loads(line))
    
    
    df = pd.DataFrame(json_list)
    
    print('Extracting entities...')
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        ent_set = set()
        for ent in list(row['entities']):
            ent = re.sub(start_tag_regex, '', ent)
            ent = re.sub(end_tag_regex, '', ent)
            ent = ent.lower()
            if '[pos]' in ent or '[neg]' in ent:
                ent = re.sub(r'\[pos\]', '', ent)
                ent = re.sub(r'\[neg\]', '', ent)
                if ent != '':
                    ent = ent.strip()
                    ent_set.add(ent)
            #ent_set.add(ent)    
        id = row['row_id']
        doc_ent_list.append(ent_set)
        id_list.append(id)
    
    
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    
    X = vectorizer.fit_transform(doc_ent_list)
    feature_names = vectorizer.get_feature_names_out()
    
    print('Normalizing entities...')
    ## Normalize disease names
    normalized_feature_names = normalize_disease(feature_names)
    
    # make dict to map feature names to new columns
    feature_dict = dict(zip(feature_names, normalized_feature_names))
    
    new_doc_ent_list = []
    
    #for doc_ent in doc_ent_list:
    for doc_ent in tqdm.tqdm(doc_ent_list, total=len(doc_ent_list)):
        new_doc_ent = set()
        for ent in doc_ent:
            if ent.lower() not in feature_dict:
                #new_doc_ent.add(ent.lower())
                print(f'{ent.lower()} not in feature_dict')
            else:
                new_doc_ent.add(feature_dict[ent.lower()])
                
        new_doc_ent_list.append(new_doc_ent)
    

    ## Frequency-based filtering
    # create matrix
    new_vectorizer = CountVectorizer(analyzer=lambda x: x)
    new_X = new_vectorizer.fit_transform(new_doc_ent_list)
    new_feature_names = new_vectorizer.get_feature_names_out()
    
    # copy new_X
    dist = new_X.copy()
    # word distribution
    dist = dist.sum(axis=0)
    dist = np.array(dist).flatten()
    
    word_freq = dict(zip(list(new_feature_names), list(dist)))
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
    
    # remove obious non-disease words
    noise_words = ['Hospital admission', 'the', 'discharge.','Hospitals']
    sorted_word_freq = {k: v for k, v in sorted_word_freq.items() if k not in noise_words}
    
    # frequency-based filtering
    
    # sum of all elements in the matrix
    total_freq = sum(sorted_word_freq.values())
    
    # get the column name until the frequency reaches 80%
    cum_freq = 0
    selected_words = []
    for word, freq in sorted_word_freq.items():
        cum_freq += freq
        if freq <= 1:
            break
        elif cum_freq/total_freq >= 0.8:
            break
        selected_words.append(word)
        
    # re-construct matrix with selected words
    selected_words_set = set(selected_words)
    final_doc_ent_list = []
    for doc_ent in new_doc_ent_list:
        # take intersection with selected_words_set
        new_doc_ent = doc_ent.intersection(selected_words_set)
        final_doc_ent_list.append(new_doc_ent)
    
    # if a value in normalized_dict is not in selected_words, replace with ''
    final_normalized_dict = {}
    for key, value in feature_dict.items():
        if value in selected_words_set:
            final_normalized_dict[key] = value
        else:
            final_normalized_dict[key] = ''
    
    with open('anonymized_data/final_normalized_dict.pkl', 'wb') as f:
        pickle.dump(final_normalized_dict, f)

    final_vectorizer = CountVectorizer(analyzer=lambda x: x)
    final_X = final_vectorizer.fit_transform(final_doc_ent_list)
    final_feature_names = final_vectorizer.get_feature_names_out()
    
    # extract only doc contains 5 to 30 words
    # get indices of docs with 5 to 30 words
    selected_indices = []
    for i, row in enumerate(final_X):
        num_words = row.sum()
        #if 5<= num_words <= 40:
        if num_words >= 10 and num_words <= 120:
            selected_indices.append(i)
    # get selected_indices from id_list
    
    selected_id_list = [id_list[i] for i in selected_indices]
    
    # sample 100 ids from selected_indices
    sampled_indices = np.random.choice(selected_indices, 100, replace=False)
    # remove sampled_indices from selected_indices
    selected_indices = [i for i in selected_indices if i not in sampled_indices]
    
    selected_id_list = [id_list[i] for i in selected_indices]
    sample_row_ids = [id_list[i] for i in sampled_indices]
    
    
    selected_X = final_X[selected_indices]
    sampled_X = final_X[sampled_indices]

    print('Anonymizing entities...')
    ## Neighbor search
    nerest_neighbors, distances, indices, nearest_neighbor_indices = find_nearest_neighbors(selected_X.toarray())
    nerest_neighbors_id_list = [selected_id_list[i] for i in nearest_neighbor_indices]

    distances_sampled, indices_sampled = find_neighbors_sampled(selected_X.toarray(), sampled_X.toarray())
    print(indices_sampled.shape)
    
    diff_mat =  selected_X.toarray() - nerest_neighbors

    # take absolute value of dff_mat
    diff_mat = np.abs(diff_mat)
    
    # change the non-zero element in dff_mat to 0 in selected_X
    anonymized_mat = selected_X.copy()
    anonymized_mat = anonymized_mat.toarray()
    #anonymized_mat[dff_filtered_mat > 0] = 0
    anonymized_mat[diff_mat > 0] = 0
    
    # to df
    anonymized_data = {}
    
    anonymized_df = pd.DataFrame(anonymized_mat, columns=final_feature_names)
    sampled_df = pd.DataFrame(sampled_X.toarray(), columns=final_feature_names)
    
    neighbor_dict = {}
    sample_neighbor_indices = indices_sampled.flatten() 
    
    pairs_dict = {}

    # iter rows and print the feature names
    for i, row in anonymized_df.iterrows():
        row_id = selected_id_list[i]
        # print non-zero column names
        anoonymized_entities = row[row > 0].index.tolist()
        anonymized_data[row_id] = anoonymized_entities
        
        # matched id
        pairs_dict[row_id] = nerest_neighbors_id_list[i]

        neighbor_dict[row_id] = sample_row_ids[sample_neighbor_indices[i]]

    sample_data = {}
    for i, row in sampled_df.iterrows():
        row_id = sample_row_ids[i]
        # print non-zero column names
        sample_entities = row[row > 0].index.tolist()
        sample_data[row_id] = sample_entities
    
    print('Saving anonymized data...')
    # save selected_id_list
    with open('anonymized_data/selected_id_list.pkl', 'wb') as f:
        pickle.dump(selected_id_list, f)

    # save nerest_neighbors_id_list
    with open('anonymized_data/nerest_neighbors_id_list.pkl', 'wb') as f:
        pickle.dump(nerest_neighbors_id_list, f)
    
    # save nearest_neighbor_indices
    with open('anonymized_data/pairs_dict.pkl', 'wb') as f:
        pickle.dump(pairs_dict, f)

    # save anonymized_data and final_normalized_dict as pkl
    with open('anonymized_data/anonymized_data.pkl', 'wb') as f:
        pickle.dump(anonymized_data, f)

    with open('anonymized_data/sampled_data.pkl', 'wb') as f:
        pickle.dump(sample_data, f)

    # save sample_neighbor_indices
    with open('anonymized_data/sample_neighbor_dict.pkl', 'wb') as f:
        pickle.dump(neighbor_dict, f)

    # save selected_X
    with open('anonymized_data/selected_X.pkl', 'wb') as f:
        pickle.dump(selected_X, f)

    # save sampled_X
    with open('anonymized_data/sampled_X.pkl', 'wb') as f:
        pickle.dump(sampled_X, f)

    # save nerest_neighbors
    with open('anonymized_data/nearest_neighbors.pkl', 'wb') as f:
        pickle.dump(nerest_neighbors, f)

    # save diff_mat
    with open('anonymized_data/diff_mat.pkl', 'wb') as f:
        pickle.dump(diff_mat, f)
    
    
if __name__ == '__main__':
    main()