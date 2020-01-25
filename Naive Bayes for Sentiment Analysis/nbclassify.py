# use this file to classify using naive-bayes classifier 
# Expected: generate nboutput.txt

import sys
import glob
import os
import collections
import re
import json
import string
import math

if __name__ == "__main__":
    source_directory=str(sys.argv[1])
    all_files = glob.glob(os.path.join(source_directory, '*/*/*/*.txt'))

    test_by_class = collections.defaultdict(list)

    for f in all_files:
        class1, class2, fold, fname = f.split('/')[-4:]
        test_by_class["test_files"].append(f)
    

# define list of stop words
# 
    stop_words=["a", "about", "above", "after", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldnt", "d", "did", "didn", "didnt", "do", "does", "doesn", "doesnt", "doing", "don", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadnt", "has", "hasn", "hasnt", "have", "haven", "havent", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isnt", "it", "its", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightnt", "more", "most", "mustn", "mustnt", "mustnt", "my", "myself", "needn", "neednt", "no", "nor", "not", "now", "o", "off", "of", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shant", "she", "shes", "should", "shouldve", "shouldn", "shouldnt", "so", "some", "such", "t", "than", "that", "thatll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasnt", "we", "were", "weren", "werent", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "wont", "would", "wouldn", "wouldnt", "wouldnt", "y", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves","through","trip","reached","my","and","itll"]


#stop_words=["a", "about", "above", "after", "ain", "all", "am", "an", "and", "any", "are", "aren", "arent", "as", "at", "be", "because", "been", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldnt", "d", "did", "didn", "didnt", "do", "does", "doesn", "doesnt", "doing", "don", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadnt", "has", "hasn", "hasnt", "have", "haven", "havent", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isnt", "it", "its", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightnt", "more", "most", "mustn", "mustnt", "mustnt", "my", "myself", "needn", "neednt", "no", "nor", "not", "now", "o", "off", "of", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shant", "she", "shes", "should", "shouldve", "shouldn", "shouldnt", "so", "some", "such", "t", "than", "that", "thatll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "we", "were", "weren", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won", "wont", "would", "wouldn", "y", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves", "through", "trip", "reached", "my", "and", "itll", "why", "which", "our", "from", "there"]

    vocabulary = json.loads(open('nbmodel.txt').read())
#print(vocabulary)
    count=0;

    open('nboutput.txt', 'w').close()

    with open("nboutput.txt", "a") as fp:
        for word in test_by_class.get("test_files"):
            data_unclassified=""
    
            with open(word, 'r') as myfile:
                data_unclassified=data_unclassified + myfile.read().replace('\n', ' ')
            data_unclassified=data_unclassified.lower()

            data_unclassified = data_unclassified.translate(string.maketrans("",""), string.punctuation)
            data_unclassified = re.sub(' +', ' ', data_unclassified).strip() 
            data_unclassified = re.sub(r'[^\w\s]','',data_unclassified)
            list_unclassified=data_unclassified.split()
            clean_list_unclassified=[]


            for term in list_unclassified:
                if term not in stop_words:
                    clean_list_unclassified.append(term)
                
            mle_class={}
        
            mle_class["positive_truthful"]=vocabulary.get("class_prior_probabilities").get("positive_truthful")
            mle_class["positive_deceptive"]=vocabulary.get("class_prior_probabilities").get("positive_deceiptful")
            mle_class["negative_truthful"]=vocabulary.get("class_prior_probabilities").get("negative truthful")
            mle_class["negative_deceptive"]=vocabulary.get("class_prior_probabilities").get("negative deceiptful")
            

        
        
            for term in clean_list_unclassified:
                if term in vocabulary:
                    mle_class["positive_truthful"] = mle_class.get("positive_truthful") * vocabulary.get(term).get("positive truthful")
                    mle_class["positive_deceptive"] = mle_class.get("positive_deceptive") * vocabulary.get(term).get("positive deceiptful")
                    mle_class["negative_truthful"] = mle_class.get("negative_truthful") * vocabulary.get(term).get("negative truthful")
                    mle_class["negative_deceptive"] = mle_class.get("negative_deceptive") * vocabulary.get(term).get("negative deceiptful")
        

            
            mle_prob=0
            
            for category in mle_class:                
                if (mle_class.get(category) > mle_prob):
                    mle_prob=mle_class.get(category)
                    assigned_class1, assigned_class2 = category.split('_')
                        
            fp.write(assigned_class2)
            fp.write(" ")
            fp.write(assigned_class1)
            fp.write(" ")
            fp.write(word)
            fp.write("\n")
