import numpy as np
import matplotlib.pyplot as plt
import Basic_Functions as bf

N = bf.HORIZON
S = bf.NUM_STATES # number of states
A = bf.NUM_ACTIONS # number of actions
# State space is {0,1, ..., S-1}
# Action space is {0,1,...,A-1}

stage_cost_c = bf.stage_cost
transition_prob = bf.T    # T(a,i,j)=Pr(xt+1=j|xt=i, ut=a)
observation_prob = bf.Z   # Z[s, o] = P(observation=o | true state=s)

# Given current belief state as input belief
# belief represents a probability distribution over the state space, 
# so belief is a vector of lengh S and has sum = 1


def bayes_update(belief,action_index,observation_index):
# This function updates a belief vector from b_t to b_{t+1} 
# when provided with new observation y_{t+1}: observation_index and new action $u_t$: action_index

   
    term1 = np.zeros(S)

    # Calculate Term1(s_next) directly from the formula
    for s_next in range(S):

        for s in range(S):
            # Please implement the belief state update here 

            # For each possible next state s_next:
            # Term1(s_next)= sum over s of belief(s)* P(s_next | s, action)* P(observation | s_next)
            # Then, the belief state b_{t+1}(s_next) = Term1(s_next) / sum over all s_next of Term1(s_next)
            
            term1[s_next] += belief[s] * transition_prob[action_index, s, s_next] * observation_prob[s_next, observation_index]



    updated_belief = np.zeros(S) # initialize b_{t+1}

    if term1.sum() > 0:
        updated_belief = (term1/term1.sum()) # normalize term1 to a prob distribution

    return updated_belief # return b_{t+1}


