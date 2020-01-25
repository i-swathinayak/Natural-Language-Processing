# use this file to learn naive-bayes classifier 
# Expected: generate nbmodel.txt

import sys
import glob
import os
import collections
import re
import json
import string

# model_file = "nbmodel.txt"
# input_path = str(sys.argv[0])

if __name__ == "__main__":
    source_directory=str(sys.argv[1])

    all_files = glob.glob(os.path.join(source_directory, '*/*/*/*.txt'))
    train_by_class = collections.defaultdict(list)

    for f in all_files:
        class1, class2, fold, fname = f.split('/')[-4:]
        train_by_class[class1+class2].append(f)

 
    total_count=0

    for word in train_by_class:
        total_count = total_count + len(train_by_class.get(word))

    total_count=float(total_count)

# compute prior probabilities of each class
    prior_prob={}

    prior_prob["positive_truthful"] = len(train_by_class.get("positive_polaritytruthful_from_TripAdvisor"))/total_count
    prior_prob["positive_deceiptful"]=len(train_by_class.get("positive_polaritydeceptive_from_MTurk"))/total_count
    prior_prob["negative_truthful"]=len(train_by_class.get("negative_polaritytruthful_from_Web"))/total_count
    prior_prob["negative_deceiptful"]=len(train_by_class.get("negative_polaritydeceptive_from_MTurk"))/total_count

# define list of stop words
#stop_words=["a", "about", "above", "after", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldnt", "d", "did", "didn", "didnt", "do", "does", "doesn", "doesnt", "doing", "don", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadnt", "has", "hasn", "hasnt", "have", "haven", "havent", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isnt", "it", "its", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightnt", "more", "most", "mustn", "mustnt", "mustnt", "my", "myself", "needn", "neednt", "no", "nor", "not", "now", "o", "off", "of", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shant", "she", "shes", "should", "shouldve", "shouldn", "shouldnt", "so", "some", "such", "t", "than", "that", "thatll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "we", "were", "weren", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won", "would", "wouldn", "y", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves","through","trip","reached","my","and","itll","why","which","our","from","there"]
  
    stop_words=["a", "about", "above", "after", "ain", "all", "am", "an", "and", "any", "are", "aren", "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldnt", "d", "did", "didn", "didnt", "do", "does", "doesn", "doesnt", "doing", "don", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadnt", "has", "hasn", "hasnt", "have", "haven", "havent", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isnt", "it", "its", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightnt", "more", "most", "mustn", "mustnt", "mustnt", "my", "myself", "needn", "neednt", "no", "nor", "not", "now", "o", "off", "of", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shant", "she", "shes", "should", "shouldve", "shouldn", "shouldnt", "so", "some", "such", "t", "than", "that", "thatll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "we", "were", "weren", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won", "would", "wouldn", "y", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves","through","trip","reached","my","and","itll","why","which","our","from","there"]

#
# create BoW for each class
#


    data_positive_truthful=""
    data_positive_deceptive=""
    data_negative_truthful=""
    data_negative_deceptive=""

    BoW_positive_truthful = {}
    BoW_positive_deceiptful = {}
    BoW_negative_truthful = {}
    BoW_negative_deceiptful = {}

#
# BoW creation for positive truthful  
#
    for word in train_by_class.get("positive_polaritytruthful_from_TripAdvisor"):
        with open(word, 'r') as myfile:
            data_positive_truthful=data_positive_truthful + myfile.read().replace('\n', ' ')

# convert the string to lowercase
    data_positive_truthful=data_positive_truthful.lower()

# remove digits and punctuations

#data_positive_truthful = re.sub(r'\d+', '', data_positive_truthful)
    data_positive_truthful = data_positive_truthful.translate(string.maketrans("",""), string.punctuation)

# remove multiple whitespaces
    data_positive_truthful=re.sub(' +', ' ', data_positive_truthful).strip() 
#data_positive_truthful=' '.join(data_positive_truthful.split())

# remove punctuations
    data_positive_truthful = re.sub(r'[^\w\s]','',data_positive_truthful)



# convert string to list
    list_positive_truthful=data_positive_truthful.split()

# declare a list to store the data without stop words
    clean_list_positive_truthful=[]
                    

    for word in list_positive_truthful:
    #if (word not in stop_words) or (not word.isdigit()):
        if (word not in stop_words):
            clean_list_positive_truthful.append(word)

                            
# tokenize
    for word in clean_list_positive_truthful:
        BoW_positive_truthful[word]=BoW_positive_truthful.get(word, 0) + 1 

    pt_word_count=len(clean_list_positive_truthful)

#
# BoW creation for positive deceptive
#
             

    for word in train_by_class.get("positive_polaritydeceptive_from_MTurk"):
        with open(word, 'r') as myfile:
            data_positive_deceptive=data_positive_deceptive + myfile.read().replace('\n', ' ')

    data_positive_deceptive=data_positive_deceptive.lower()
#data_positive_deceptive=' '.join(data_positive_deceptive.split())
# remove digits and punctuations
#data_positive_deceptive = re.sub(r'\d+', '', data_positive_deceptive)
    data_positive_deceptive = data_positive_deceptive.translate(string.maketrans("",""), string.punctuation)

    data_positive_deceptive = re.sub(' +', ' ', data_positive_deceptive).strip() 
    data_positive_deceptive = re.sub(r'[^\w\s]','',data_positive_deceptive)

    list_positive_deceptive=data_positive_deceptive.split()
    clean_list_positive_deceptive=[]

    for word in list_positive_deceptive:
        if (word not in stop_words):
            clean_list_positive_deceptive.append(word)

                                                
    for word in clean_list_positive_deceptive:
        BoW_positive_deceiptful[word]=BoW_positive_deceiptful.get(word, 0) + 1 

    pd_word_count=len(clean_list_positive_deceptive)

#
# BoW creation for negative truthful
#
                                                    
    for word in train_by_class.get("negative_polaritytruthful_from_Web"):
        with open(word, 'r') as myfile:
            data_negative_truthful=data_negative_truthful + myfile.read().replace('\n', ' ')

    data_negative_truthful=data_negative_truthful.lower()
#data_negative_truthful=' '.join(data_negative_truthful.split())
#data_negative_truthful = re.sub(r'\d+', '', data_negative_truthful)
    data_negative_truthful = data_negative_truthful.translate(string.maketrans("",""), string.punctuation)

    data_negative_truthful=re.sub(' +', ' ', data_negative_truthful).strip() 
    data_negative_truthful = re.sub(r'[^\w\s]','',data_negative_truthful)

    list_negative_truthful=data_negative_truthful.split()
    clean_list_negative_truthful=[]

    for word in list_negative_truthful:
        if (word not in stop_words):
            clean_list_negative_truthful.append(word)

                                                                   
    for word in clean_list_negative_truthful:
        BoW_negative_truthful[word]=BoW_negative_truthful.get(word, 0) + 1 

    nt_word_count=len(clean_list_negative_truthful)

#
# BoW creation for negative deceptive
#
                            
    for word in train_by_class.get("negative_polaritydeceptive_from_MTurk"):
        with open(word, 'r') as myfile:
            data_negative_deceptive=data_negative_deceptive + myfile.read().replace('\n', ' ')

    data_negative_deceptive=data_negative_deceptive.lower()
#data_negative_deceptive=' '.join(data_negative_deceptive.split())
#data_negative_deceptive = re.sub(r'\d+', '', data_negative_deceptive)
    data_negative_deceptive = data_negative_deceptive.translate(string.maketrans("",""), string.punctuation)

    data_negative_deceptive=re.sub(' +', ' ', data_negative_deceptive).strip() 
    data_negative_deceptive = re.sub(r'[^\w\s]','',data_negative_deceptive)

    list_negative_deceptive=data_negative_deceptive.split()
    clean_list_negative_deceptive=[]

    for word in list_negative_deceptive:
        if (word not in stop_words):
            clean_list_negative_deceptive.append(word)

    for word in clean_list_negative_deceptive:
        BoW_negative_deceiptful[word]=BoW_negative_deceiptful.get(word, 0) + 1 

    nd_word_count=len(clean_list_negative_deceptive)

# 
# |V|
#
    vocabulary={}

    for word in BoW_positive_truthful: 
        vocabulary[word]=1;

    for word in BoW_positive_deceiptful: 
        vocabulary[word]=1;

    for word in BoW_negative_truthful: 
        vocabulary[word]=1;

    for word in BoW_negative_deceiptful: 
        vocabulary[word]=1;

                                                                                                           

# compute the probability of each word conditioned on every class
    term_prob={}
    B=float(len(vocabulary))
    term_prob["class_prior_probabilities"]={"positive_truthful" : prior_prob["positive_truthful"], "positive_deceiptful" : prior_prob["positive_deceiptful"],
                        "negative truthful" : prior_prob["negative_truthful"], "negative deceiptful" : prior_prob["negative_deceiptful"] }

    for word in vocabulary:
        term_prob[word]={"positive truthful": (BoW_positive_truthful.get(word,0) + 1)/(pt_word_count + B), "positive deceiptful" : 
                      (BoW_positive_deceiptful.get(word,0) + 1)/(pd_word_count + B), 
                    "negative truthful": (BoW_negative_truthful.get(word,0) + 1)/(nt_word_count + B),
                    "negative deceiptful": (BoW_negative_deceiptful.get(word,0) + 1)/(nd_word_count + B)}

                                                                                                               
        
    open('nbmodel.txt', 'w').close()

    with open("nbmodel.txt", "w+") as fp:
        fp.write(json.dumps(term_prob, indent=2))


