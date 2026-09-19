import numpy as np
import matplotlib.pyplot as plt
import Basic_Functions as bf

N = bf.HORIZON
S = bf.NUM_STATES # number of states
A = bf.NUM_ACTIONS # number of actions
stage_cost_c = bf.stage_cost
transition_prob = bf.T    # T(a,i,j)=Pr(xt+1=j|xt=i, ut=a)
observation_prob = bf.Z   # Z[s, o] = P(observation=o | true state=s)



def choose_action_QMDP(belief,t,J_opt):

    # --------------------------------------------------------
    # QMDP policy:
    #
    # For each action a, calculate
    #
    #   sum over s of belief(s) *
    #
    #       [c(s,a)+ sum over s_next of P(s_next | s,a) J_(t+1)(s_next)]
    # Then choose the action with minimum belief-weighted expected cost.
    # --------------------------------------------------------

    action_costs = []
    for a in range(A):

        belief_weighted_cost = 0.0

        for s in range(S):

            # Immediate cost c(s,a)
            cost_if_state_s = stage_cost_c(s,a)

            # Expected future cost from state s
            expected_future_cost = 0.0

            for s_next in range(S):

                expected_future_cost += (transition_prob[a,s,s_next]*J_opt[t + 1,s_next])

            cost_if_state_s += expected_future_cost

            # Weight this state's action cost by belief(s)
            belief_weighted_cost += (belief[s]*cost_if_state_s)

        action_costs.append(belief_weighted_cost )

    best_action = np.argmin(action_costs)

    return best_action

