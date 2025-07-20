import os
import json
import pandas as pd
import re

# need a script to convert generated data from all methods to json
# all data should be in the format {'row_id': row_id, 'generated_text': generated_text}
# baselines might not include prompt in the generation results

save_path = 'data/generated_corpus.json'

def main():
    results_dir = 'generation_results/'
    path_list = os.listdir(results_dir)

    # open json file
    generation_results_dict = {}
    for path in path_list:
        row_id = path.replace('.json', '')
        with open(results_dir + path, 'r') as f:
            obj = json.load(f)
            generation_results_dict[row_id] = obj

    
    
    for row_id in generation_results_dict.keys():
        
        gen_result = generation_results_dict[row_id]
        promt = gen_result['prompt']
        
    
        generate_text = gen_result['generated_text'].replace(promt, '')
        gen_lines = generate_text.split('\n')
    
        all_str_list = []
        # get string inbetween ||
        for line in gen_lines:
            text= line.split('|')
            if len(text) == 1:
                continue
            else:
                text = text[-2]
        
            if text == '' or text == ' ':
                pass
            # elif text is number
            # use regex
            elif re.match(r'^-?\d+(?:\.\d+)?$', text):
                pass
            else:
                all_str_list.append(text)
                
        cleaned_text = '\n'.join(all_str_list)

        data_point = {'row_id': row_id, 'generated_text': cleaned_text}
        # write to json
        with open(save_path, 'a') as f:
            json.dump(data_point, f)
            f.write('\n')

if __name__ == "__main__":
    main()