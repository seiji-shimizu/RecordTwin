import pandas as pd
import json
import pickle   
import re
import tqdm

mimic_note_path = '../_data/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz'
json_path = '../NEextraction/sectioned_entity_dict_negated.json'
#save_dir = 'disease_fewshot_data.json'

save_dir = 'sectioned_negated_fewshot_data.json'
section_name_set = ['admission  date:', 'service:', 'allergies:', 'chief  complaint:', 'major  surgical  or  invasive  procedure:', 'history  of  present  illness:', 'past  medical  history:', 'past  surgical  history:', 'allergies:', 'medications  on  admission:', 'physical  examination:', 'physical  exam:', 'hospital  course:', 'discharge  diagnosis:', 'discharge  medications:', 'discharge  instructions:', 'follow-up  instructions:', 'follow-up  plan:', 'follow-up  medications:', 'follow-up  diagnosis:', 'medications:', 'initial  laboratory  studies:', 'partnent  results:', 'family  history:', 'social  history:']
section_name_set = ["~"+i+"~" for i in section_name_set]

def main():
    
    d_anonymize = False
    test = True
    
    if d_anonymize:
        dict_path = '../Anonymization/disease_final_normalized_dict.pkl'
        anonymized_data_path = '../Anonymization/disease_anonymized_data.pkl'
    elif test:  
        dict_path = '../Anonymization/anonymized_data/negated_test_final_normalized_dict.pkl'
        anonymized_data_path = '../Anonymization/anonymized_data/negated_test_anonymized_data.pkl'
        sample_data_path = '../Anonymization/anonymized_data/negated_test_sampled_data.pkl'
        sample_neighborhood_path = '../Anonymization/anonymized_data/negated_test_sample_neighbor_dict.pkl'
        d_anonymize = True
    else:
        dict_path = '../Anonymization/final_normalized_dict.pkl'
        anonymized_data_path = '../Anonymization/anonymized_data.pkl'
    
    # print path name
    print(dict_path)
    print(anonymized_data_path)

    # read pkl files
    with open(dict_path, 'rb') as f:
        normalized_dict = pickle.load(f)
    
    with open(anonymized_data_path, 'rb') as f:
        anonymized_data = pickle.load(f)

    if test:
        with open(sample_data_path, 'rb') as f:
            sample_data = pickle.load(f)
        with open(sample_neighborhood_path, 'rb') as f:
            sample_neighborhood_dict = pickle.load(f)

    mimic_df = pd.read_csv(mimic_note_path)

    # sample some rows
    #sampled_rows = mimic_df.sample(1000)
    #row_ids = list(set(list(sampled_rows['ROW_ID'].values)))
    
    json_list = []
    
    with open(json_path, 'r') as f:
        # read lines
        lines = f.readlines()
        # each line is a json object
        for line in lines:
            json_list.append(json.loads(line))
    
    # change here for k-anonymized version
    
    start_tag_regex = re.compile(r'<e\d+>')
    end_tag_regex = re.compile(r'</e\d+>')
    
    
    
    #for row in json_list:
    # tqdm
    for row in tqdm.tqdm(json_list, total=len(json_list)):
        data_point = []
        anonymized_data_point = []
        row_id = row['row_id']
        entities = row['entities']

        

        if (row_id not in anonymized_data) and (row_id not in sample_data):
            continue

        if row_id in anonymized_data:
            anonymized_ent = set(anonymized_data[row_id])
            sample = False
            neigbhor = sample_neighborhood_dict[row_id]

        elif row_id in sample_data:
            anonymized_ent = set(sample_data[row_id])
            sample = True
            neigbhor = None

        for ent in entities:
            # remove tags
            if ent in section_name_set:
                data_point.append(ent)
                anonymized_data_point.append(ent)
                continue

            ent = start_tag_regex.sub('', ent)
            ent = end_tag_regex.sub('', ent)
            ent = ent.lower()

            ent = ent.strip()
            data_point.append(ent)
            
            ent_key = ent
            
            pos = False
            neg = False
            # if ent contains str [pos]
            if '[pos]' in ent_key:
                pos = True
                # remove [pos]
                ent_key = ent_key.replace('[pos]', '')
                
            if '[neg]' in ent_key:
                neg = True
                ent_key = ent_key.replace('[neg]', '')
            
            ent_key = ent_key.strip()

            if d_anonymize:
                if pos or neg:
                    if ent_key in normalized_dict.keys():
                        normalized_ent = normalized_dict[ent_key]
                        # if not deleted add to data point
                        if normalized_ent in anonymized_ent:
                            if pos:
                                normalized_ent = normalized_ent + ' [pos]'
                            if neg:
                                normalized_ent = normalized_ent+ ' [neg]'
        
                            anonymized_data_point.append(normalized_ent)
                            
                # if not disease name simply add to data point
                else:
                    anonymized_data_point.append(ent)
                    #print(anonymized_data_point)
                    
 
                        
            else:
                if ent_key in normalized_dict:
                    normalized_ent = normalized_dict[ent_key]
                    if normalized_ent in anonymized_ent:
                        if pos:
                            normalized_ent = normalized_ent.replace('problemtagged', '[pos]')
                        if neg:
                            normalized_ent = normalized_ent.replace('problemtagged', '[neg]')
    
                        anonymized_data_point.append(normalized_ent)

                    
            
        text = mimic_df[mimic_df['ROW_ID'] == int(row_id)].iloc[0]['TEXT']
        json_obj = {'row_id': row_id, 'entities': data_point, 'anonymized_entities': anonymized_data_point, 'text': text, 'sample': sample, 'neighbor': neigbhor}
        with open(save_dir, 'a') as f:
            f.write(json.dumps(json_obj) + '\n')

        # to df and save as pkl file

    json_list = []

    with open(save_dir, 'r') as f:
        # read lines
        lines = f.readlines()
        for line in lines:
            json_list.append(json.loads(line))

    save_df = pd.DataFrame(json_list)
    save_df.to_pickle('sectioned_negated_fewshot_data.pkl')

         
if __name__ == "__main__":
    main()