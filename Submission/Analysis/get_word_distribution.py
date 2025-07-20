import pickle
import gensim
from gensim.corpora import Dictionary
from nltk.corpus import stopwords

# Download the stopwords once
import nltk
nltk.download('stopwords')

    
cls_dir = '../Datasets/'
icd_dir = cls_dir+'icd_classification/'
phenotyping_dir = cls_dir+'phenotyping/'
readmission_dir = cls_dir+'Readmission/'
org_dir = '0/'
gen_dir = '1/'
train_path = 'train.pkl'

icd_org_train = pickle.load(open(icd_dir+org_dir+train_path, 'rb'))
icd_gen_train = pickle.load(open(icd_dir+gen_dir+train_path, 'rb'))
phenotyping_org_train = pickle.load(open(phenotyping_dir+org_dir+train_path, 'rb'))
phenotyping_gen_train = pickle.load(open(phenotyping_dir+gen_dir+train_path, 'rb'))
readmission_org_train = pickle.load(open(readmission_dir+org_dir+train_path, 'rb'))
readmission_gen_train = pickle.load(open(readmission_dir+gen_dir+train_path, 'rb'))


import gensim
from gensim.corpora import Dictionary

def get_class_dict(org_train, gen_train):
    
    class_dict_org = {}
    
    for data in org_train:
        labels = data['labels']
        text = data['text'].split()[:1000]
        text = ' '.join(text)
        # if labels is a list iterate
        if isinstance(labels, list):
            for i in range(len(labels)):
                if i not in class_dict_org:
                    class_dict_org[i] = []  # Initialize the list if the label is new
                    if labels[i] == 1:
                        class_dict_org[i].append(text)
                else:
                    if labels[i] == 1:
                        class_dict_org[i].append(text)
        else:
            if labels not in class_dict_org:
                class_dict_org[labels] = []  # Initialize the list if the label is new
            class_dict_org[labels].append(text)  # Append the text regardless

    class_dict_gen = {}

    for data in gen_train:
        labels = data['labels']
        text = data['text']
        # if labels is a list iterate
        if isinstance(labels, list):
            for i in range(len(labels)):
                if i not in class_dict_gen:
                    class_dict_gen[i] = []
                    if labels[i] == 1:
                        class_dict_gen[i].append(text)
                else:
                    if labels[i] == 1:
                        class_dict_gen[i].append(text)
        else:   
            if labels not in class_dict_gen:
                class_dict_gen[labels] = []  # Initialize the list if the label is new
            class_dict_gen[labels].append(text)  # Append the text regardless
    
    return class_dict_org, class_dict_gen

def get_bow(class_dict):
    bow_dict = {}
    dict_dict = {}
    
    # Load the stop words
    stop_words = set(stopwords.words('english'))  # Change 'english' to your preferred language

    for cls in class_dict:
        if cls not in bow_dict:
            bow_dict[cls] = []
        
        # Tokenize and filter out stop words
        tokenized_texts = [
            [word for word in gensim.utils.simple_preprocess(text) if word not in stop_words] 
            for text in class_dict[cls]
        ]
        
        dictionary = Dictionary(tokenized_texts)
        dict_dict[cls] = dictionary
        dictionary.filter_extremes()
        
        all_texts = []
        for text in tokenized_texts:
            all_texts += text
        
        bow = dictionary.doc2bow(all_texts)
        bow_dict[cls] = bow

    return bow_dict, dict_dict

def main():
    icd_org_class_dict, icd_gen_class_dict = get_class_dict(icd_org_train, icd_gen_train)
    phenotyping_org_class_dict, phenotyping_gen_class_dict = get_class_dict(phenotyping_org_train, phenotyping_gen_train)
    readmission_org_class_dict, readmission_gen_class_dict = get_class_dict(readmission_org_train, readmission_gen_train)

    icd_org_bow_dict, icd_org_dict_dict = get_bow(icd_org_class_dict)
    icd_gen_bow_dict, icd_gen_dict_dict = get_bow(icd_gen_class_dict)
    phenotyping_org_bow_dict, phenotyping_org_dict_dict = get_bow(phenotyping_org_class_dict)
    phenotyping_gen_bow_dict, phenotyping_gen_dict_dict = get_bow(phenotyping_gen_class_dict)
    readmission_org_bow_dict, readmission_org_dict_dict = get_bow(readmission_org_class_dict)
    readmission_gen_bow_dict, readmission_gen_dict_dict = get_bow(readmission_gen_class_dict)

    # save the dictionaries
    save_dir = 'word_distribution/'
    pickle.dump(icd_org_bow_dict, open(save_dir+'icd_org_bow_dict.pkl', 'wb'))
    pickle.dump(icd_gen_bow_dict, open(save_dir+'icd_gen_bow_dict.pkl', 'wb'))
    pickle.dump(icd_org_dict_dict, open(save_dir+'icd_org_dict_dict.pkl', 'wb'))
    pickle.dump(icd_gen_dict_dict, open(save_dir+'icd_gen_dict_dict.pkl', 'wb'))

    pickle.dump(phenotyping_org_bow_dict, open(save_dir+'phenotyping_org_bow_dict.pkl', 'wb'))
    pickle.dump(phenotyping_gen_bow_dict, open(save_dir+'phenotyping_gen_bow_dict.pkl', 'wb'))
    pickle.dump(phenotyping_org_dict_dict, open(save_dir+'phenotyping_org_dict_dict.pkl', 'wb'))
    pickle.dump(phenotyping_gen_dict_dict, open(save_dir+'phenotyping_gen_dict_dict.pkl', 'wb'))

    pickle.dump(readmission_org_bow_dict, open(save_dir+'readmission_org_bow_dict.pkl', 'wb'))
    pickle.dump(readmission_gen_bow_dict, open(save_dir+'readmission_gen_bow_dict.pkl', 'wb'))
    pickle.dump(readmission_org_dict_dict, open(save_dir+'readmission_org_dict_dict.pkl', 'wb'))
    pickle.dump(readmission_gen_dict_dict, open(save_dir+'readmission_gen_dict_dict.pkl', 'wb'))

    
if __name__ == '__main__':
    main()