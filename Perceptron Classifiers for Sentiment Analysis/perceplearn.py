# use this file to learn perceptron classifier
# Expected: generate vanillamodel.txt and averagemodel.txt

import sys
import glob
import os
import collections
import re
import json
import time
import math
from heapq import nlargest
import numpy as np
import random

# global variables
# define list of stop words
stop_words = ["a", "aaa", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
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
              "thatll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
              "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn",
              "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom",
              "why", "will", "with", "won", "won't", "would", "wouldn", "wouldn't", "wouldnt", "y", "you", "youd",
              "youll",
              "youre", "youve", "your", "yours", "yourself", "yourselves", "suggest", "room", "hotel", "us", "could",
              "place", "person", "made", "also", "trip", "men",
              "person", "given", "l", "ago", "along", "gets", "il", "youre", "youd", "w", "nd", "ft", "theres"]

pos_neg_list = []
tru_dec_list = []
features_list=[]
pos_word_count = 0
neg_word_count = 0



def parse_training_data(source_directory):
    ## List all files, given the root of training data.
    all_files = glob.glob(os.path.join(source_directory, '*/*/*/*.txt'))
    train_by_class = collections.defaultdict(list)
    N = 0
    for f in all_files:
        N += 1;
        class1, class2, fold, fname = f.split('/')[-4:]
        train_by_class[class1 + class2].append(f)
    return train_by_class, float(N);


def preprocess_pos_neg_data(source1, source2, dict_word_count, vocabulary, tf_dict, word_tf):
    class_word_count=0
    source = [source1, source2]
    for fold in source:
        for doc in train_by_class.get(fold):
            with open(doc, 'r') as myfile:
                file_string = myfile.read().replace('\n', ' ').lower();
                file_string = ' '.join(file_string.split());
                file_string = re.sub(r'[^\w\s]', '', file_string);
                file_string = re.sub('\\d', '', file_string)
                file_content = file_string.split();
                cleaned_list = []
                doc_dict = {}
                doc_length = 0;
                for word in file_content:
                    if word not in stop_words:
                        if word not in doc_dict:
                            vocabulary[word] = vocabulary.get(word, 0) + 1;
                        cleaned_list.append(word)
                        doc_dict[word] = doc_dict.get(word, 0) + 1;
                        doc_length += 1;
                        dict_word_count[word] = dict_word_count.get(word, 0) + 1;
                        word_tf[word] = word_tf.get(word, 0) + 1;
                        class_word_count+=1

                for word in doc_dict:
                    if word in tf_dict:
                        tf_dict[word].append(doc_dict.get(word, 0) / float(doc_length));
                    else:
                        tf_dict[word] = [doc_dict.get(word, 0) / float(doc_length)];

                pos_neg_list.append(cleaned_list);
    return class_word_count


def preprocess_tru_dec_data(source1, source2):
    source = [source1, source2]
    for fold in source:
        for doc in train_by_class.get(fold):
            with open(doc, 'r') as myfile:
                file_string = myfile.read().replace('\n', ' ').lower();
                file_string = ' '.join(file_string.split());
                file_string = re.sub(r'[^\w\s]', '', file_string);
                file_string = re.sub('\\d', '', file_string)
                file_content = file_string.split();
                cleaned_list = []
                doc_dict = {}
                doc_length = 0;
                for word in file_content:
                    if word not in stop_words:
                        cleaned_list.append(word)
                tru_dec_list.append(cleaned_list);


def compute_tf(words_tf, c_word_count):
    for word in words_tf:
        words_tf[word] = words_tf.get(word, 0) / float(c_word_count)

def compute_tf_idf(vocabulary, tf_dict, dict_word_count, N, tf_dict1, tf_dict2):
    tfidf_dict = {}
    for word in vocabulary:
        vocabulary[word] = math.log10(N / float(dict_word_count.get(word)))
    for word in tf_dict:
        #tfidf_dict[word] = vocabulary.get(word) * (sum(tf_dict.get(word)) / float(len(tf_dict.get(word))))
        tfidf_dict[word] = vocabulary.get(word) * ((tf_dict1.get(word, 0) + tf_dict2.get(word, 0))/float(2))
    return tfidf_dict;


def compute_features(tfidf_dict):
    top_features = nlargest(2000, tfidf_dict, key=tfidf_dict.get)
    return top_features;


def remove_low_freq_word(tfidf_dict):
    for word in dict_word_count:
        if dict_word_count[word] < 3:
            tfidf_dict.pop(word)
    # print(len(dict_word_count))
    # print(len(tfidf_dict))


def compute_word_frequency(document):
    word_freq_dict = {}
    for word in document:
        word_freq_dict[word] = word_freq_dict.get(word, 0) + 1;
    return word_freq_dict


def generate_feature_vector(document):
    feature_vector = np.zeros(2000)
    word_freq = compute_word_frequency(document)
    count = 0
    for word in features_list:
        if word in word_freq:
            feature_vector[count] = word_freq.get(word)
        count += 1
    return feature_vector


def generate_feature_matrix(training_list):
    X_local = np.zeros(shape=(1, 2000))
    for doc in training_list:
        feature_vector = generate_feature_vector(doc)
        feature_vector = feature_vector.reshape((1, 2000))
        X_local = np.concatenate((X_local, feature_vector), axis=0)
    X_local = X_local[1:, :]
    return X_local


def train_vanilla_perceptron(X, y, features_count, iterations_count):
    # print("Executing Vanilla Perceptron ...")
    X = np.where(X > 0, 1, 0)
    weight = np.zeros(shape=(features_count,))
    bias = 0
    #random.seed(50)
    for i in range(iterations_count):
        num = random.randint(0, len(X) - 1)
        activation = np.dot(weight, X[num]) + bias
        if y[num] * activation <= 0:
            weight = np.sum((weight, y[num] * X[num]), axis=0)
            bias = bias + y[num]
    return weight, bias


def train_averaged_perceptron(X, y, features_count, iterations_count):
    # print("Executing Averaged Perceptron ...")
    X = np.where(X > 0, 1, 0)
    weight = np.zeros(shape=(features_count,))
    c_weight = np.zeros(shape=(features_count,))
    bias = 0
    c_bias = 0
    count = 1
    #random.seed(50)
    for i in range(iterations_count):
        num = random.randint(0, len(X) - 1)
        activation = np.dot(weight, X[num]) + bias
        if y[num] * activation <= 0:
            weight = np.sum((weight, y[num] * X[num]), axis=0)
            bias = bias + y[num]
            c_weight = np.sum((c_weight, y[num] * X[num] * count), axis=0)
            c_bias = c_bias + (y[num] * count)
        count += 1
    inv_count = 1 / count
    return np.subtract(weight, inv_count * c_weight), bias - (inv_count * c_bias)


def generate_model(classifier1, classifier2, vanilla_weight_pos_neg, vanilla_bias_pos_neg, vanilla_weight_tru_dec,
                   vanilla_bias_tru_dec):
    model_dict = dict()
    model_dict[classifier1 + "_bias"] = float(vanilla_bias_pos_neg[0])
    model_dict[classifier2 + "_bias"] = float(vanilla_bias_tru_dec[0])
    features_pos_neg = vanilla_weight_pos_neg.tolist()
    features_classifier_1 = dict(zip(features_list, features_pos_neg))
    model_dict[classifier1] = features_classifier_1;
    features_tru_dec = vanilla_weight_tru_dec.tolist()
    features_classifier_2 = dict(zip(features_list, features_tru_dec))
    model_dict[classifier2] = features_classifier_2;
    return model_dict


if __name__ == "__main__":
    model_file = "vanillamodel.txt"
    avg_model_file = "averagemodel.txt"

    #input_path = str(sys.argv[1])
    start_time = time.time()
    source_directory = str(sys.argv[1])
    #source_directory = "/Users/swathinayak/Documents/op_spam_training_data"
    train_by_class, doc_count = parse_training_data(source_directory);

    dict_word_count = {}
    pn_vocabulary = {}
    pn_tf_dict = {}

    pos_word_tf = {}
    neg_word_tf = {}
    pos_word_count = preprocess_pos_neg_data("positive_polaritytruthful_from_TripAdvisor", "positive_polaritydeceptive_from_MTurk",
                            dict_word_count, pn_vocabulary, pn_tf_dict, pos_word_tf);
    neg_word_count = preprocess_pos_neg_data("negative_polaritytruthful_from_Web", "negative_polaritydeceptive_from_MTurk",
                            dict_word_count, pn_vocabulary, pn_tf_dict, neg_word_tf);

    compute_tf(pos_word_tf, pos_word_count)
    compute_tf(neg_word_tf, neg_word_count)

    unique_word_count = len(dict_word_count)
    pn_tfidf_dict = compute_tf_idf(pn_vocabulary, pn_tf_dict, dict_word_count, unique_word_count, pos_word_tf, neg_word_tf)
    remove_low_freq_word(pn_tfidf_dict)
    features_list = compute_features(pn_tfidf_dict)
    #features_list.sort()
    #print(features_list)

    preprocess_tru_dec_data("positive_polaritytruthful_from_TripAdvisor", "negative_polaritytruthful_from_Web");
    preprocess_tru_dec_data("positive_polaritydeceptive_from_MTurk", "negative_polaritydeceptive_from_MTurk");

    # print("--- %s seconds ---" % (time.time() - start_time))

    # compute feature vectors
    X_pos_neg = generate_feature_matrix(pos_neg_list)
    X_tru_dec = generate_feature_matrix(tru_dec_list)

    # print("--- %s seconds ---" % (time.time() - start_time))

    length = len(X_pos_neg)
    y = list()
    y[:length] = [1] * length
    y = np.array(y)
    index = int(length / 2)
    y[index:length] = -1
    y = y.reshape((length, 1))

    vanilla_weight_pos_neg, vanilla_bias_pos_neg = train_vanilla_perceptron(X_pos_neg, y, 2000, 4000)
    vanilla_weight_tru_dec, vanilla_bias_tru_dec = train_vanilla_perceptron(X_tru_dec, y, 2000, 4000)

    # print(vanilla_weight_pos_neg)
    # print(vanilla_weight_tru_dec)

    averaged_weight_pos_neg, averaged_bias_pos_neg = train_averaged_perceptron(X_pos_neg, y, 2000, 4000)
    averaged_weight_tru_dec, averaged_bias_tru_dec = train_averaged_perceptron(X_tru_dec, y, 2000, 4000)

    # print(averaged_weight_pos_neg)
    # print(averaged_weight_tru_dec)

    classifier1 = "positive_negative"
    classifier2 = "truthful_deceiptful"

    open(model_file, 'w').close()
    open(avg_model_file, 'w').close()

    vanilla_model = generate_model(classifier1, classifier2, vanilla_weight_pos_neg, vanilla_bias_pos_neg,
                                   vanilla_weight_tru_dec, vanilla_bias_tru_dec);
    with open(model_file, 'w') as fp:
        fp.write(json.dumps(vanilla_model, indent=2))

    averaged_model = generate_model(classifier1, classifier2, averaged_weight_pos_neg, averaged_bias_pos_neg,
                                    averaged_weight_tru_dec, averaged_bias_tru_dec);
    
    with open(avg_model_file, 'w') as fp:
        fp.write(json.dumps(averaged_model, indent=2))

    #print("--- %s seconds ---" % (time.time() - start_time))



    #print(features_list)
