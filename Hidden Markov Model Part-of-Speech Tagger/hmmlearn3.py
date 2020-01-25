import sys
import json
from heapq import nlargest

def read_input(train_file):
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as myfile:
        for line in myfile:
            train_data.append(line)
    return train_data

def compute_transition_count(train_data):
    transition_count = {}
    tag_count = {}
    line_count = 0
    for line in train_data:
        line_count += 1;
        data = line.split()
        m = len(data)
        for i in range(0, m):
            tag_list = data[i].rsplit('/',1)            
            tag = tag_list[1]
            transition_count[tag] = transition_count.get(tag, 0) + 1
    tag_count=dict(transition_count)
    top_tags = nlargest(8, tag_count, key=tag_count.get)
    top_tag={}
    count=8
    for tag in top_tags:
        top_tag[tag] = count
        count-=1
    transition_count["q0"] = line_count
    tag_count["q1"] = 0
    return transition_count, tag_count, top_tag
    

def compute_transition_prob(train_data, transition_count, tag_count):
    transition_prob = {}
    for tag in transition_count:
        transition_prob[tag] = {}   
    trans_count = 0
    for line in train_data:
        data = line.split()
        m = len(data)
        trans_count = m + 2
        start = data[0].rsplit('/',1)
        transition_prob["q0"][start[1]] = transition_prob["q0"].get(start[1], 0) + 1;
        for i in range(0, m - 1):
            s = data[i].rsplit('/',1)
            t = data[i + 1].rsplit('/',1)
            transition_prob[s[1]][t[1]] = transition_prob[s[1]].get(t[1], 0) + 1;
        end = data[m-1].rsplit('/',1)
        transition_prob[end[1]]["q1"] = transition_prob[end[1]].get("q1", 0) + 1;
    # calculate the transitional probabilities
    for key in transition_prob:
        for tag in transition_prob[key]:
            transition_prob[key][tag] = (transition_prob[key].get(tag) + 1) / (float(transition_count.get(key) + trans_count))
    # transitional probability smoothing
    for tag in transition_prob:
        for key in tag_count:
            if key not in transition_prob[tag]:
                transition_prob[tag][key] = 1 / (float(transition_count.get(tag, 0) + trans_count))  
    return transition_prob

def compute_emission_prob(train_data, tag_count):
    emission_prob = {}
    for line in train_data:
        data = line.split()
        m = len(data)
        for i in range(0, m):
            s = data[i].rsplit('/',1)
            tag = s[1]
            word=s[0]
            if word not in emission_prob:
                emission_prob[word] = {}
            emission_prob[word][tag] = emission_prob[word].get(tag, 0) + 1;
    for key in emission_prob:
        for tag in emission_prob[key]:
            emission_prob[key][tag] = emission_prob[key][tag] / float(tag_count.get(tag))            
    return emission_prob

if __name__=="__main__":
    model_file="hmmmodel.txt"
    train_file = sys.argv[1]
    train_data = read_input(train_file)    
    # calculate the transition and occurrence counts for each tag
    transition_count, tag_count, top_tag = compute_transition_count(train_data)
    # store the transitions in a dictionary and calculate transitional probabilities
    transition_prob = compute_transition_prob(train_data, transition_count, tag_count)    
    # calculate the emission probabilities
    emission_prob = compute_emission_prob(train_data, tag_count)
    # generate model
    model_dict = dict()
    model_dict["Transitional_probabilities"] = transition_prob;
    model_dict["Emission_probabilities "] = emission_prob;
    model_dict["Most_frequent_tags"] = top_tag
    with open(model_file, 'w') as fp:
        fp.write(json.dumps(model_dict, indent=2))

    