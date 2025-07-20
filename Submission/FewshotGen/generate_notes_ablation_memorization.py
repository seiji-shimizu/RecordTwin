import torch
from torch import cuda,bfloat16
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers

import pandas as pd
import tqdm
import json
import os
import pickle


from utils import generate_prompt

# seed everything

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

save_dir = 'generation_results_ablation_memorization2/'
# if directory does not exist create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


tokenizer = AutoTokenizer.from_pretrained(model_id)

template_path = "templates/sample1.txt"
template = open(template_path, "r").read()

ablation_dir = "generation_results_ablation/"
ablation_files = os.listdir(ablation_dir)
ablation_files = [i for i in ablation_files if i.endswith('.json')]
id_list = [i.replace('.json','') for i in ablation_files]

import random
random.seed(0)
random.shuffle(id_list)
print(len(id_list))

longformer_tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer')

def reformat_text_entities(text, entities, anonymized):
    
    ents = [i if i != '<br>' else '\n' for i in entities]
    anonymized = [i if i != '<br>' else '\n' for i in anonymized]
    #text = text.split('\n')
    ents = ','.join(ents)
    anonymized = ','.join(anonymized)

    return text, ents, anonymized

def shorten_text_entities(text_lines, entities_lines, max_length=1100):
    
    shortened_text = ''
    shortened_entities = ''
    #text_len = 0
    for i, (text_line, entities_line) in enumerate(zip(text_lines, entities_lines)):
        if len(longformer_tokenizer(shortened_text + text_line, return_tensors="pt")['input_ids'][0])< max_length:
            shortened_text += text_line + '\n'
            shortened_entities += entities_line + '\n'
        else:
            break
    

    # if there is no remaining text return None
    if i == len(text_lines) - 1:
        return shortened_text, shortened_entities
    else:
        remining_text = text_lines[i:]
        remaining_entities = entities_lines[i:]
    
    return shortened_text, shortened_entities

def main():
    
    data_path = 'data/sectioned_negated_fewshot_data.pkl'
    df = pd.read_pickle(data_path)
    
    sample_df = df[df['sample'] == True]
    data_df = df[df['sample'] == False]

    # get the text length for each row by splitting the text by '\n' and counting the number of lines
    sample_df['text_length'] = sample_df['text'].apply(lambda x: len(x.split('\n')))
    data_df['text_length'] = data_df['text'].apply(lambda x: len(x.split('\n')))
    
    generated_ids = [int(i.replace('.json','')) for i in os.listdir(save_dir)]

    # make sure column 'row_id' is int
    data_df['row_id'] = data_df['row_id'].astype(int)
    
    print(len(data_df))
    # drop df where row_id is in generated_ids
    data_df = data_df[~data_df['row_id'].isin(generated_ids)]
    print(len(data_df))
    
    quant_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
         device_map='auto'
    )
    
    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        repetition_penalty=1.1
    )
    
    # iter over df
    # tqdm
    
    id_dict = data_df.set_index('row_id').to_dict('index')

    for row_id in tqdm.tqdm(id_list, total=len(id_list)):
        row_id = int(row_id)
        # Fetch the row from the dictionary using row_id
        if row_id in id_dict:
            row = id_dict[row_id]
            text = row['text']
            entities = row['entities']
            anonymized = row['anonymized_entities']
            
            reformated_text, reformated_entities, reformated_anonymized = reformat_text_entities(text, entities, anonymized)
            
            #########
            text_lines = reformated_text.split("\n")
            entity_lines = reformated_entities.split("\n")
            a_entity_lines = reformated_anonymized.split("\n")

            empty_indices = [i for i, text in enumerate(text_lines) if not text]
            
            # remove empty lines from text_lines and entity_lines based on empty_indices
            new_text_lines = []
            new_entity_lines = []
            new_a_entity_lines = []
            for i, text in enumerate(text_lines):
                if i not in empty_indices:
                    new_text_lines.append(text)
                    new_entity_lines.append(entity_lines[i])
                    new_a_entity_lines.append(a_entity_lines[i])
        
            text_lines = new_text_lines
            entity_lines = new_entity_lines
            a_entity_lines = new_a_entity_lines
            
            #########

            shortened_text, shortened_entities = shorten_text_entities(text_lines, entity_lines, 1100)
            shortened_text, shortened_anonymized = shorten_text_entities(text_lines, a_entity_lines, 1100)
            
            data = [{'text': shortened_text, 'entities': shortened_entities}]
            prompt = generate_prompt(template, data, shortened_anonymized)
            
            
            prompt_len = len(tokenizer(prompt)['input_ids'])
            print(prompt_len)
            max_length = prompt_len + 1500
        else:
            continue
            
        print("Generating text")
        try:
            sequences = pipeline(
                    prompt,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=5,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=max_length,
                )
        except Exception as e:
            print(e)
            continue

        generation_result_path = save_dir + f'{row_id}.json'
        
        
        json_obj = {'generated_text': sequences[0]['generated_text'], 'original_text': shortened_text, 'prompt': prompt, 'entities': shortened_entities}
        with open(generation_result_path, 'w') as f:
            json.dump(json_obj, f)
            
        



if __name__ == '__main__':
    main()  
    #CUDA_VISIBLE_DEVICES=0,1,3 python generate_notes_ablation_memorization.py

#scp -r generation_results_ablation_memorization seiji-sh@asagi.naist.jp:/home/is/seiji-sh/TwinRecord/_data/