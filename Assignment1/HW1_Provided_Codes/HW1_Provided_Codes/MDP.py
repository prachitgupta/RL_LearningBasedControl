import numpy as np
import matplotlib.pyplot as plt
import Basic_Functions as bf


N = bf.HORIZON # horizon length
S = bf.NUM_STATES # number of states
A = bf.NUM_ACTIONS # number of actions
# State space is {0,1, ..., S-1}
# Action space is {0,1,...,A-1}
stage_cost_c = bf.stage_cost # stage cost function: stage_cost_c(x,u)
transition_prob = bf.T    # transition prob: transition_prob(a,i,j)=Pr(xt+1=j|xt=i, ut=a)





# ============================================================
# Dynamic Programming will calculate J_t(s) for t=0,..., N, and all states s
# ============================================================

# Use J_opt(t,x) to store the optimal cost-to-go function J_t^*(x) for t=0,..., N and all states x
J_opt = np.zeros((N + 1, S))

# At t=N
J_opt[N, :] = bf.terminal_costs

for t in range(N-1,-1,-1): # backward induction
    for s_index in range(S):

        # Please implement dynamic programming here
        # J_t(s) = min over actions of {c(s,a)+sum over s_next P(s_next | s,a) J_(t+1)(s_next)}

        J_opt[t,s_index] = min([stage_cost_c(s_index, a) + sum([transition_prob[a, s_index, s_next] * J_opt[t + 1, s_next] for s_next in range(S)]) for a in range(A)])






if __name__ == "__main__":
    bf.show_J(0,J_opt) # Plot J_0^* in a heat map
    bf.show_J(6,J_opt) # Plot J_6^* in a heat map
    plt.show()