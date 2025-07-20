import csv
import pickle
import random
import os

# seed
seed = 42
# set seed
random.seed(seed)

path = 'anonymized_data/pairs_dict.pkl'

with open(path, 'rb') as f:
    pairs_dict = pickle.load(f)

generated_dir = '../_data/generation_results/'
id_list = [int(i.replace('.json','')) for i in os.listdir(generated_dir)]
id_set = set(id_list)

# extract key and value both in id_set
pairs_dict = {k: v for k, v in pairs_dict.items() if int(k) in id_set and int(v) in id_set}

# load sampled_pairs.csv and only sampled from unsampled pairs
path = 'sampled_pairs.csv'
with open(path, 'r') as f:
    reader = csv.reader(f)
    sampled_pairs = list(reader)
    sampled_pairs = [(int(i[0]), int(i[1])) for i in sampled_pairs]

pairs_list = [(int(k), int(v)) for k, v in pairs_dict.items()]
pairs_list = [i for i in pairs_list if i not in sampled_pairs]

# sample 1000 pairs
sampled_pairs = random.sample(pairs_list, 1000)


# write to csv
path = 'sampled_pairs2.csv'
with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    for pair in sampled_pairs:
        writer.writerow(pair)