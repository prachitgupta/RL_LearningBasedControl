import numpy as np
import matplotlib.pyplot as plt
import Basic_Functions as bf

N = bf.HORIZON
S = bf.NUM_STATES # number of states
A = bf.NUM_ACTIONS # number of actions
stage_cost_c = bf.stage_cost
transition_prob = bf.T    # T(a,i,j)=Pr(xt+1=j|xt=i, ut=a)
observation_prob = bf.Z   # Z[s, o] = P(observation=o | true state=s)





def choose_action_MAP(belief,t,J_opt):
# Input is current belief b_t, current stage t, 
# and the optimal cost to go from the fully observable case: J_opt. 
# J_opt[t,s] = the optimal cost to go J_t^*(s)

    
    most_likely_state = np.argmax(belief)

    # Please implement the belief state update here 
    # best action = argmin_a {c(s_MAP,a)+sum over s_next of P(s_next | s_MAP,a) J_opt(t+1)(s_next)}

    best_action = np.argmin([stage_cost_c(most_likely_state, a) + sum([transition_prob[a, most_likely_state, s_next] * J_opt[t + 1, s_next] for s_next in range(S)]) for a in range(A)])

    return best_action

