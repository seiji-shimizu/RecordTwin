import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import json
from tqdm import tqdm
import os
import pickle
import re

#sample_num = 200
max_length = 350

mimic_note_path = '../_data/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz'

task_name = 'i2b2_2012'

label_list_2010 = ['O', 'B-treatment', 'I-treatment', 'B-problem', 'I-problem', 'B-test', 'I-test']
label_list_2012 = ['B-OCCURRENCE', 'O', 'B-TREATMENT', 'I-OCCURRENCE', 'B-PROBLEM', 'I-PROBLEM', 'B-CLINICAL_DEPT', 'I-CLINICAL_DEPT', 'B-TEST', 'I-TEST', 'B-EVIDENTIAL', 'I-TREATMENT', 'I-EVIDENTIAL']

#model_path = '../fine-tunning/outputs/pytorch_model.bin'
#model_path = '../output/150/i2b2_2012/seed_1/pytorch_model.bin'
#model_path = '../DownstreamTasks/_token_results/i2b2_2012_text/experiment_1/pytorch_model.bin'
model_path = 'fine_tunning/output/350/i2b2_2012/seed_1/pytorch_model.bin'
#model_name = 'yikuan8/Clinical-Longformer'
model_name = 'emilyalsentzer/Bio_ClinicalBERT'

def get_label_list(label_list):
        unique_labels = set(label_list)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list


def preprocess_text(text):
    # split into words
    # first with \n
    # and space
    sentences = text.split('\n')
    words = []
    #print(f'num_sentences: {len(sentences)}')
    for sentence in sentences:
        word_list = sentence.split(' ')
        # remove ''
        #word_list = [word for word in word_list if word]
        words.extend(word_list)
        words.append('\n')
    return words

def convert_doc_to_samples(json_obj, tokenizer, max_length):
    
    # clinical longformer: '\n' is a token
    # clinical bert: '\n' is not a token
    max_length = max_length - 2
    
    chunk_dict = {}
    word_list = json_obj['text']
    
    
    if 'labels' in json_obj.keys():
        label_list = json_obj['labels']
    else:
        label_list = None
    if 'id' in json_obj.keys():
        doc_id = json_obj['id']
    
    # if '\n' is not in the tokenizer vocab, add it
    if '\n' not in tokenizer.get_vocab():
        tokenizer.add_tokens('\n')
    
    tokenized_inputs = tokenizer(word_list, padding='do_not_pad', max_length=None, truncation=False, is_split_into_words=True)
        
    input_ids = tokenized_inputs['input_ids']
    word_ids = tokenized_inputs.word_ids()
    
    # revmove first and last word_ids and input_ids since they are special tokens
    word_ids = word_ids[1:-1]
    input_ids = input_ids[1:-1]
    id_chunks = []
    line_break_id = tokenizer.get_vocab()['\n']
    
    # combine sentences unless total length of tokens < max_length
    line_break_indices = [i for i, x in enumerate(input_ids) if x == line_break_id]
    #print(f'line_break_indices: {len(line_break_indices)}')
    
    start_index = 0
    previous_line_break_index = -1
    
    for i, line_break_index in enumerate(line_break_indices):
        
        if len(input_ids) < max_length:
            id_chunks.append((0, len(input_ids)-1))
            break
        
        elif (line_break_index - start_index)+1 < max_length:
            previous_line_break_index = line_break_index
            continue

        elif (line_break_index - start_index)+1 > max_length and (previous_line_break_index  - start_index)+1 <= max_length:
            id_chunks.append((start_index, previous_line_break_index))
            start_index = previous_line_break_index+1
            previous_line_break_index = line_break_index

        else:
            id_chunks.append((start_index, line_break_index))
            start_index = line_break_index+1
            previous_line_break_index = line_break_index
    
    # After loop, add any remaining tokens as a final chunk
    if start_index < len(input_ids):
        id_chunks.append((start_index, len(input_ids) - 1))
    
    line_break_sum = 0

    for i, id_chunk in enumerate(id_chunks):
        # get the start and end of the chunk
        input_start_id = id_chunk[0]
        input_end_id = id_chunk[1]
        
        # get the corresponding word start and end index
        word_start_idx = word_ids[input_start_id]
        word_end_idx = word_ids[input_end_id]
        
        word_chunk = word_list[word_start_idx:word_end_idx+1]
        
        # tokenize word_chunk and print the input_ids length
        tokenized_chunk = tokenizer(word_chunk, padding='do_not_pad', max_length=None, truncation=False, is_split_into_words=True)
        input_ids_chunk = tokenized_chunk['input_ids']
        
        if len(input_ids_chunk) > max_length+2:
            pass
        if label_list:
            label_chunk = label_list[word_start_idx:word_end_idx+1]
        else:
            label_chunk = None
        
        if 'id' in json_obj.keys():
            doc_id = json_obj['id']
        else:
            doc_id = None
        
        line_break_indices = [i for i, x in enumerate(word_chunk) if x == '\n']
        line_break_sum += len(line_break_indices)

        chunk_dict[i] = {'text': word_chunk, 'labels': label_chunk, 'doc_id': doc_id}

    (f'line_break_sum: {line_break_sum}')
    return chunk_dict



def predict_labels(model, tokenizer, chunks, max_len, label_list):
    
    all_word_label_pairs = {}
    
    line_break_sum = 0

    for idx, chunk in chunks.items():
        # Tokenize
        chunk_words = chunk['text']
        
        tokenized_inputs = tokenizer(chunk_words, padding='max_length', max_length=max_len, truncation=True, is_split_into_words=True)
        input_ids = tokenized_inputs['input_ids']
        word_ids = tokenized_inputs.word_ids()

        # Convert to tensor
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to('cuda')

        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(input_ids).logits

        # Get the predicted labels
        predicted_label_ids = torch.argmax(logits, dim=2)
        
        word_label_pairs = {}
        
        for word_id in set(word_ids):
            if word_id is None:
                continue
            
            word = chunk_words[word_id]

            word_id_indices = [i for i, wid in enumerate(word_ids) if wid == word_id]
            label_id = predicted_label_ids[0][word_id_indices[0]].item()  # Use the first occurrence for the label
            
            label = label_list[label_id]
            
            word_label_pairs[word_id] = (word, label)
        
        # insert line break
        line_break_indices = [i for i, x in enumerate(chunk_words) if x == '\n']

        line_break_sum += len(line_break_indices)
        
        for line_break_index in line_break_indices:
            word_label_pairs[line_break_index] = ('\n', 'O')
        
        # sort by word_id
        word_label_pairs = dict(sorted(word_label_pairs.items()))
        
        all_word_label_pairs[idx] = word_label_pairs
    
    #print(f'line_break_sum: {line_break_sum}')

    return all_word_label_pairs

def main():
    
    # read as df
    df = pd.read_csv(mimic_note_path)#, nrows=sample_num)

    if task_name == 'i2b2_2012':
       num_labels = len(label_list_2012)
       label_list = get_label_list(label_list_2012)
       label_to_id = {label: i for i, label in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label_to_id,
            id2label={i: l for l, i in label_to_id.items()},
            finetuning_task='ner',
        )    
    print('\n====loading model====')
    # load model
    # to cuda
    #on cpu
    #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    d_summaries = df[df.CATEGORY == 'Discharge summary']
    #for i, row in d_summaries.iterrows():
    # tqdm
    #count = 0
    #for i, row in tqdm(d_summaries.iterrows(), total=sample_num):
    # all rows
    for i, row in tqdm(d_summaries.iterrows(), total=d_summaries.shape[0]):
        
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        # Wrap the model with DataParallel
        #if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model)

        model.to('cuda')
        
    
        summary = row.TEXT
        row_id = row.ROW_ID
        # preprocess text
        words = preprocess_text(summary)
        # number of \n in words
        num_line_breaks = words.count('\n')
        
        # convert to samples
        samples = convert_doc_to_samples({'text': words}, AutoTokenizer.from_pretrained(model_name), max_length)
        # predict labels
        all_word_label_pairs = predict_labels(model, AutoTokenizer.from_pretrained(model_name), samples, max_len=max_length, label_list=label_list)
        # save
        save_dir = 'word_label_pairs/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{row_id}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(all_word_label_pairs, f)
        
if __name__ == '__main__':
        main()