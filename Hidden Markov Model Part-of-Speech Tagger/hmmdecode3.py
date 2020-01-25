import sys
import json
import numpy as np
from heapq import nlargest
import time

#start_time = time.time()

#reload(sys)
#sys.setdefaultencoding('utf-8')
# global variables
global transition_matrix
global emission_matrix
global tags_enum
global words_enum
#np.set_printoptions(threshold=sys.maxsize)

def compute_te_matrices(input_file, model_file):
    global transition_matrix
    global emission_matrix
    global tags_enum
    global words_enum
    transition_emission_prob = json.loads(open(model_file).read())
    transition_prob = transition_emission_prob["Transitional_probabilities"]
    emission_prob = transition_emission_prob["Emission_probabilities "]
    top_tags = transition_emission_prob["Most_frequent_tags"]

    state_count=len(transition_prob)-1
    obs_count=len(emission_prob)
    
    transition_matrix = np.zeros((state_count,state_count+1))
    start_state = np.zeros((state_count,))    
    tags_enum={}
    count=0
    for tag in transition_prob:
        if ( tag != "q0" ):
            tags_enum[tag] = count
            start_state[count] = transition_prob["q0"].get(tag)
            count+=1
    transition_prob.pop("q0")
    tags_enum["q1"] = count
    for tag in transition_prob:
        for key in transition_prob[tag]:
            transition_matrix[tags_enum.get(tag)][tags_enum.get(key)] = transition_prob[tag].get(key)
     
    #print(transition_matrix)
    #print(tags_enum)
    emission_matrix = np.zeros((state_count,obs_count))
    words_enum={}
    count=0
    for word in emission_prob:
        words_enum[word] = count
        count+=1
    for word in emission_prob:
        for tag in emission_prob[word]:
            emission_matrix[tags_enum.get(tag)][words_enum.get(word)] = emission_prob[word].get(tag)
            
    top_tag = nlargest(8, top_tags, key=top_tags.get)
    val = 0.9
    for tag in top_tag:
        top_tags[tags_enum.get(tag)] = val
        val = val - 0.1
            
    return start_state, state_count, top_tags

def viterbi_decode(input_file, output_file, start_state, state_count, top_tags):
    open(output_file, 'w').close()
    with open(output_file, "a", encoding='utf-8') as fp:
        with open(input_file, 'r', encoding='utf-8') as myfile:
            for line in myfile:
                data = line.split()
                obs_count=len(data)
                decode, backtrack = np.zeros((state_count,obs_count)), np.zeros((state_count,obs_count), dtype=np.int)               
                backtrack[0] = 0;

                #if data[0] in words_enum:
                    #decode[:,0] = start_state.T * emission_matrix[:,words_enum[data[0]]]
                #else:
                    #decode[:,0] = start_state.T
                 
                 
                start_state = start_state.reshape((state_count, 1))
                   

                for s in range (0,state_count):
                    if data[0] in words_enum:
                        decode[s,0] = start_state[s] * emission_matrix[s,words_enum[data[0]]]
                    elif s in top_tags:
                        decode[s,0] = start_state[s] * top_tags.get(s)
                    else:
                        decode[s,0] = start_state[s] * 0.1
                        
                for t in range(1,obs_count): 
                    for s in range (0,state_count): 
                        prev_prob = decode[:,t-1] * transition_matrix[:,s]
                        decode[s,t] = np.amax(prev_prob)
                        backtrack[s,t] = np.argmax(prev_prob)
                        if data[t] in words_enum:
                            decode[s,t] = decode[s,t] * emission_matrix[s,words_enum[data[t]]]
                        elif s in top_tags:
                            decode[s,t] = decode[s,t] * top_tags.get(s)
                        else:
                            decode[s,t] = decode[s,t] * 0.1
                            
                    if t == obs_count-1:
                        decode[:,t] = decode[:,t] * transition_matrix[:,tags_enum["q1"]]
                            
                result = np.zeros(obs_count, dtype=np.int); 
                result[obs_count-1] =  decode[:,obs_count-1].argmax()
                for t in range(obs_count-1,0,-1): 
                    result[t-1] = backtrack[result[t],t]
            
                count=0
                for idx, j in enumerate(result):
                    for tag, value in tags_enum.items():    
                        if value == int(j):
                            data[count] = data[count] + "/" + tag
                            count+=1
                        
                decode_data = ' '.join(data)
                fp.write(decode_data)
                fp.write('\n')
                
                #print(decode)
                
if __name__=="__main__":
    output_file="hmmoutput.txt"
    model_file="hmmmodel.txt"
    input_file = sys.argv[1]
    # create transition and emission matrices
    start_state, state_count, top_tags = compute_te_matrices(input_file, model_file)       
    # implement viterbi decoding
    viterbi_decode(input_file, output_file, start_state, state_count, top_tags)
    #print("--- %s seconds ---" % (time.time() - start_time))





                     



    
