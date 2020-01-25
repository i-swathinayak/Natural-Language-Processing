import sys
import glob
import os
import collections
import re
import json

# global variables
stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
              "aren't", "as", "at", "be", "because", "been",
              "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did",
              "didn", "didn't", "do", "does", "doesn", "doesn't", "doing",
              "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has",
              "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her",
              "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't",
              "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn",
              "mightn't", "more", "most", "mustn", "mustn't", "mustnt", "my", "myself", "needn", "needn't", "no", "nor",
              "not", "now", "o", "off", "of", "on", "once", "only", "or", "other", "our", "ours",
              "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should",
              "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that",
              "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
              "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn",
              "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom",
              "why", "will", "with", "won", "won't", "would", "wouldn", "wouldn't", "wouldnt", "y", "you", "you'd",
              "you'll",
              "you're", "you've", "your", "yours", "yourself", "yourselves", "suggest", "room", "hotel", "us", "could",
              "place", "person", "made", "also", "trip",
              "person", "given", "l", "ago", "along", "gets", "il", "youre", "youd", "w", "nd", "ft","theres"]

def parse_testing_data(source_directory):
    all_files = glob.glob(os.path.join(source_directory, '*/*/*/*.txt'))
    test_by_class = collections.defaultdict(list)
    for f in all_files:
        class1, class2, fold, fname = f.split('/')[-4:]
        test_by_class["test_files"].append(f)
    return test_by_class
 

def compute_word_frequency(document):
    word_freq_dict = {}
    for word in document:
        word_freq_dict[word] = word_freq_dict.get(word, 0) + 1;
    return word_freq_dict
            
if __name__ == "__main__":
    model_file = str(sys.argv[1])
    output_file = "percepoutput.txt"
    #input_path = str(sys.argv[2])
    # all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
    #input_path = str(sys.argv[0])
    source_directory = str(sys.argv[2])
    #source_directory="/Users/swathinayak/Documents/op_spam_training_data"
    test_by_class = parse_testing_data(source_directory)

    model_features = json.loads(open(model_file).read())

    open(output_file, 'w').close()

    for doc in test_by_class.get("test_files"):
        data_unclassified = ""

        with open(doc, 'r') as myfile:
            data_unclassified = myfile.read().replace('\n', ' ')
            data_unclassified = data_unclassified.lower()
            data_unclassified = ' '.join(data_unclassified.split())
            data_unclassified = re.sub(r'[^\w\s]', '', data_unclassified)
            data_unclassified = re.sub('\\d', '', data_unclassified)
            list_unclassified = data_unclassified.split()
            clean_list_unclassified = []

            for term in list_unclassified:
                if term not in stop_words:
                    clean_list_unclassified.append(term)

            word_count={}
            word_count = compute_word_frequency(clean_list_unclassified)
            pn_activation=0
            for word in word_count:
            #for word in clean_list_unclassified:
                if word in model_features["positive_negative"]:
                    #pn_activation = pn_activation + (word_count.get(word) * model_features["positive_negative"].get(word));
                    pn_activation = pn_activation + (model_features["positive_negative"].get(word));
            pn_activation = pn_activation + model_features.get("positive_negative_bias")

            if (pn_activation >= 0):
                assigned_class_2 = "positive"
            else:
                assigned_class_2 = "negative"


            td_activation=0
            for word in word_count:
            # for word in clean_list_unclassified:
                if word in model_features["truthful_deceiptful"]:
                    #td_activation = td_activation + (word_count.get(word) * model_features["truthful_deceiptful"].get(word));
                    td_activation = td_activation + (model_features["truthful_deceiptful"].get(word));
            td_activation = td_activation + model_features.get("truthful_deceiptful_bias")

            if (td_activation >= 0):
                assigned_class_1 = "truthful"
            else:
                assigned_class_1 = "deceptive"

            with open(output_file, "a") as fp:
                fp.write(assigned_class_1)
                fp.write(" ")
                fp.write(assigned_class_2)
                fp.write(" ")
                fp.write(doc)
                fp.write("\n")


