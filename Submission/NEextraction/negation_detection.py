import re
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import os
import pickle


model_path_negatoin = '../DownstreamTasks/_cls_results/i2b2_2012_status/experiment_1/pytorch_model.bin'
model_name = 'emilyalsentzer/Bio_ClinicalBERT'
entity_save_dir = 'entity_dict_negated.json'


neg_max_length = 240



def get_surrounding_text(text, size=100):
    words = text.split()
    # identify <e> and </e> indices
    start = -1
    end = -1
    for i, word in enumerate(words):
        if word == '<e>':
            start = i
        elif word == '</e>':
            end = i

    if start == -1 or end == -1:
        return None
    # get surrounding text
    start = max(0, start - size)
    end = min(len(text), end + size)
    words_list = words[start:end]
    return ' '.join(words_list)

def return_negation_data(marked, entity_list):
    all_marked_data = []
    regex_ent = re.compile(r'<e\d+>.*?</e\d+>')
    id_entity_list = regex_ent.findall(' '.join(marked))
    id_ent_tuple_list = []
    # sort by id, <eid>entity</eid>
    for ent in id_entity_list:
        id_regex = re.compile(r'<e(\d+)>')
        id = int(id_regex.findall(ent)[0])
        id_ent_tuple_list.append((int(id), ent))
    # sort by id
    id_ent_tuple_list.sort(key=lambda x: x[0])
    
    linebreak_indices = [i for i, e_type in enumerate(entity_list) if e_type == '<br>']
    for linebreak_index in linebreak_indices:
        id_ent_tuple_list.insert(linebreak_index, ('<br>', '<br>'))
    
    
    all_entity_and_types = []
    for i , (id, ent) in enumerate(id_ent_tuple_list):
        if id == '<br>':
            all_entity_and_types.append(('<br>', '<br>', '<br>'))
        else:
            all_entity_and_types.append((f'<e{id}>', ent, entity_list[i]))
    
    
    target_type = {'PROBLEM'}
    for i, entity_type in enumerate(entity_list):
        # only keep f'<e{i}>' and f'</e{i}>' and remove other tags
        if entity_type not in target_type:
            continue
        remove_tags = [f'<e{j}>' for j in range(len(entity_list)) if j != i] + [f'</e{j}>' for j in range(len(entity_list)) if j != i]
        datapoint = [word for word in marked if word not in remove_tags]
        datapoint = ' '.join(datapoint)
        entity_id = f'<e{i}>'
        datapoint = datapoint.replace(f'<e{i}>', ' <e>')
        datapoint = datapoint.replace(f'</e{i}>', ' </e>')
        if '<e>' in datapoint:
            datapoint = get_surrounding_text(datapoint)
            all_marked_data.append((datapoint, entity_id, True))
    
    return all_marked_data, all_entity_and_types


def return_marked_sentences(all_word_label_pairs):
    words_list = []
    labeld_list = []

    for id, dict in all_word_label_pairs.items():
        for word_id, (word, label) in dict.items():
            words_list.append(word)
            labeld_list.append(label)

    marked = []
    entity_type_list = []
    
    all_aligned = []
    
    for i, (word, label) in enumerate(zip(words_list, labeld_list)):
        all_aligned.append((word, label))
    
    entity_id = 0
    pointer = 0
    while pointer < len(all_aligned):
        if all_aligned[pointer][1] == 'O' and all_aligned[pointer][0] != '\n':
            marked.append(all_aligned[pointer][0])
            pointer += 1
        elif all_aligned[pointer][1][0] == 'B':
            entity_type = all_aligned[pointer][1][2:]
            entity_type_list.append(entity_type)
            marked.append(f'<e{entity_id}>')
            marked.append(all_aligned[pointer][0])
            pointer += 1
            while pointer < len(all_aligned) and all_aligned[pointer][1][0] == 'I':
                marked.append(all_aligned[pointer][0])
                pointer += 1
            marked.append(f'</e{entity_id}>')
            entity_id += 1
        elif all_aligned[pointer][0] == '\n':
            entity_type_list.append('<br>')
            pointer += 1
        else:
            marked.append(all_aligned[pointer][0])
            pointer += 1
    return marked, entity_type_list


negation_labe2id = {'POS': 0, 'NEG': 1}

def main():

    
    negation_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            label2id=negation_labe2id,
            id2label={i: l for l, i in negation_labe2id.items()},
            problem_type='single_label_classification'
         )
    
    negation_id2label = negation_model.config.id2label
    #if torch.cuda.device_count() > 1:
    #    negation_model = torch.nn.DataParallel(negation_model)
    negation_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add <e> and </e> to tokenizer
    negation_tokenizer.add_tokens(['<e>', '</e>'])
    # add <e> and </e> to model
    negation_model.resize_token_embeddings(len(negation_tokenizer))
    negation_state_dict = torch.load(model_path_negatoin, map_location=torch.device('cpu'))
    negation_model.load_state_dict(negation_state_dict)

    negation_model.to('cuda')

    negation_model.eval()
    
    folder = 'word_label_pairs'
    done_id_list = []
    
    if os.path.exists(entity_save_dir):
        # read as lines of json
        with open(entity_save_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                json_obj = json.loads(line)
                done_id_list.append(json_obj['row_id'])

    done_path_list = [os.path.join('word_label_pairs', f'{file}.pkl') for file in done_id_list]

    # list of all path
    all_word_label_pairs_list = [os.path.join(folder, file) for file in os.listdir(folder)]
    print(len(set(all_word_label_pairs_list) - set(done_path_list)))
    all_word_label_pairs_list = list(set(all_word_label_pairs_list) - set(done_path_list))
    #for all_word_label_pairs_path in all_word_label_pairs_list:
    # tqdm
    for all_word_label_pairs_path in tqdm.tqdm(all_word_label_pairs_list, total=len(all_word_label_pairs_list)):
        # read pickle
        with open(all_word_label_pairs_path, 'rb') as f:
            all_word_label_pairs = pickle.load(f)
        row_id = all_word_label_pairs_path.replace('.pkl', '').replace('word_label_pairs/', '')
        
        ### Negation prediction ###

        marked, entity_list = return_marked_sentences(all_word_label_pairs)
        negation_data, all_entities = return_negation_data(marked, entity_list)
        #all_negation_data.append({"row_id": row_id, "negation_data": negation_data, "entities": all_entities})
        
        
        negation_data_list = negation_data
        entities = all_entities
        
        disease_per_doc = {}
        
        # batch size = 4
        batched_marked_text = []
        batched_entity_id = []
        batched_marked = []
        for i, (marked_text, entity_id, marked) in enumerate(negation_data_list):
            batched_marked_text.append(marked_text)
            batched_entity_id.append(entity_id)
            batched_marked.append(marked)
            if len(batched_marked_text) == 16 or i == len(negation_data_list)-1:
                with torch.no_grad():
                    tokenized_inputs = negation_tokenizer(batched_marked_text, padding='max_length', max_length=neg_max_length, truncation=True)
                    input_ids = tokenized_inputs['input_ids']
                    input_ids = torch.tensor(input_ids).to('cuda')
                    logits = negation_model(input_ids).logits
                    predicted_label_ids = torch.argmax(logits, dim=1)
                    
                    for j, predicted_label_id in enumerate(predicted_label_ids):
                        
                        #label = negation_model.config.id2label[predicted_label_id.item()]
                        label = negation_id2label[predicted_label_id.item()]
                        entity_id = batched_entity_id[j]
                        marked = batched_marked[j]
                        marked_text = batched_marked_text[j]

                        if marked == True:
                            # get <e>entity</e> from marked
                            entity = re.search(r'<e>.*?</e>', marked_text).group()
                            # replace <e>entity</e> with <e{id}>entity</e{id}>
                            entity_id_int = int(entity_id[2:-1])
                            entity = entity.replace('<e>', f'<e{entity_id_int}>')
                            entity = entity.replace('</e>', f' [{label}] </e{entity_id_int}>')
                            disease_per_doc[entity_id] =  entity
                batched_marked_text = []
                batched_entity_id = []
                batched_marked = []
                
        all_entities_data = []
        for k, (entity_id, entity, entity_type) in enumerate(entities):
            if entity_type == '<br>':
                all_entities_data.append('<br>')
            elif entity_id in disease_per_doc:
                all_entities_data.append(disease_per_doc[entity_id])
            else:
                all_entities_data.append(entity)
              
        

        json_obj = {"row_id":row_id}
        json_obj["entities"] = all_entities_data
        # save
        with open(entity_save_dir, 'a') as f:
            f.write(json.dumps(json_obj)+'\n')

        #count += 1
        #if count == sample_num:
        #    break

if __name__ == '__main__':
    main()