import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap




# ============================================================
# Grid, Actions, and Feasible/Valid State Space
# ============================================================

WIDTH = 6  # Grid is {0,1,2,..., 5} X {0,1,2,..., 5}
HEIGHT = 6
HORIZON = 15 # T=15

goal = (5, 5) # The goal is upper right corner.

obstacles = {(2, 2),(2, 3),(3, 3)}

action_names = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT"
]

actions = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0)
}

NUM_ACTIONS = len(action_names)



# if (x,y) is inside the grid and doesn't coincide with any obstacle,
# then it is valid.

def valid(position):
    x, y = position

    return (
        0 <= x < WIDTH
        and 0 <= y < HEIGHT
        and position not in obstacles
    )






states = []  # add all valid states to this list

for y in range(HEIGHT):
    for x in range(WIDTH):
        if valid((x, y)):
            states.append((x, y))


NUM_STATES = len(states)


# Define a 1-dim index for the two-dim states.
state_to_index = {
    state: i
    for i, state in enumerate(states)
}  

index_to_state = {
    i: state
    for i, state in enumerate(states)
}

goal_index = state_to_index[goal]

# print("Number of valid states:", NUM_STATES)




# ============================================================
# Transition Probabilities:
# Two rules: 
# Rule 1: If wall or obstacle is hit, stay where you are and return the original state
# Rule 2: follow intended direction with prob 0.8, and slip towards two orthogonal directions with each prob 0.1
# ============================================================


# Rule 1: If wall or obstacle is hit, stay where you are and return the original state
def next_position(state, action_name):
    # Goal is absorbing
    if state == goal:
        return goal

    x, y = state

    dx, dy = actions[action_name]

    candidate = (
        x + dx,
        y + dy
    )

    if valid(candidate):
        return candidate

    return state


# ============================================================
# Rule 2:
# Intended direction:     0.8
# Side slip:              0.1
# The other side slip:    0.1
# ============================================================

def actual_action_probabilities(commanded_action):

    if commanded_action == "UP":
        return [
            ("UP", 0.8),
            ("LEFT", 0.1),
            ("RIGHT", 0.1)
        ]

    elif commanded_action == "DOWN":
        return [
            ("DOWN", 0.8),
            ("RIGHT", 0.1),
            ("LEFT", 0.1)
        ]

    elif commanded_action == "LEFT":
        return [
            ("LEFT", 0.8),
            ("DOWN", 0.1),
            ("UP", 0.1)
        ]

    elif commanded_action == "RIGHT":
        return [
            ("RIGHT", 0.8),
            ("UP", 0.1),
            ("DOWN", 0.1)
        ]

    else:
        raise ValueError("Unknown action")



# Transition Probabilities take into account of both Rule 1 and Rule 2
T = np.zeros(
    (
        NUM_ACTIONS,
        NUM_STATES,
        NUM_STATES
    )
)


for a_index, commanded_action in enumerate(action_names):

    for s_index, state in enumerate(states):

        # Goal is absorbing
        if state == goal:

            T[
                a_index,
                s_index,
                s_index
            ] = 1.0

            continue

        outcomes = actual_action_probabilities(
            commanded_action
        ) # outcomes is a prob distribution of the possible actions to be taken

        for actual_action, probability in outcomes:

            new_state = next_position(
                state,
                actual_action
            )

            new_index = state_to_index[
                new_state
            ]

            T[
                a_index,
                s_index,
                new_index
            ] += probability   # The += accounts for the probability of staying still due to bumping into obstacles or walls.



# Verify that the transition probability matrices have row sum =1
# print(
#     "Transition probability row sums:",
#     T.sum(axis=2).min(),
#     T.sum(axis=2).max()
# )



# ============================================================
# Cost functions:
# Every non-goal step costs 1.
# Goal costs 0.
# Terminal Cost = 15 if not reaching goal
# ============================================================

def stage_cost(state_index, action_index):

    state = index_to_state[state_index]

    if state == goal:
        return 0.0

    return 1.0


def terminal_cost(state):

    if state == goal:
        return 0.0
    else:
        return 15.0

terminal_costs = np.array([
    terminal_cost(state)
    for state in states
])




# ============================================================
# Visualization of J_t at a heatmpa
# ============================================================

def show_J(t,J_opt):
    grid = np.full(
        (
            HEIGHT,
            WIDTH
        ),
        np.nan
    )


    # --------------------------------------------
    # Fill grid with J_t values
    # --------------------------------------------

    for s_index, state in enumerate(states):
        x, y = state
        grid[y,x] = J_opt[t,s_index]


    # --------------------------------------------
    # Plot heatmap
    # --------------------------------------------

    plt.figure()

    cmap = LinearSegmentedColormap.from_list("lighter",plt.cm.YlOrBr_r(np.linspace(0.3, 1, 256)))

    plt.imshow(grid, origin="lower", cmap=cmap)


    plt.colorbar(label="Cost-to-go")


    # --------------------------------------------
    # Write exact J_t value inside each state
    # --------------------------------------------

    for s_index, state in enumerate(states):

        x, y = state

        value = J_opt[t,s_index]


        if state == goal:

            text = (f"G\n" f"{value:.2f}")

        else:

            text = ( f"{value:.2f}" )


        plt.text(x,y,text,ha="center",va="center")


    # --------------------------------------------
    # Mark obstacles
    # --------------------------------------------

    for x, y in obstacles:

        plt.text(x,y,"X",ha="center",va="center")


    # --------------------------------------------
    # Axis labels
    # --------------------------------------------

    plt.xticks(range(WIDTH ))

    plt.yticks(range(HEIGHT))

    plt.xlabel("x")

    plt.ylabel("y")

    plt.title(f"Cost-to-go J_{t}")








# ============================================================
# Observation Model
#
# Correct observation: w.p. 0.80
# Error UP:            w.p. 0.05
# Error DOWN:          w.p. 0.05
# Error LEFT:          w.p. 0.05
# Error RIGHT:         w.p. 0.05
#
# If an observation is invalid,
# its probability is added back to the true location.
# ============================================================



# Z[s, o] = P(observation=o | true state=s)
Z = np.zeros((NUM_STATES,NUM_STATES))


observation_errors = [((0, 0), 0.80),
    ((0, 1), 0.05),
    ((0, -1), 0.05),
    ((-1, 0), 0.05),
    ((1, 0), 0.05)]


for s_index, state in enumerate(states):

    x, y = state
    for (dx, dy), probability in observation_errors:

        candidate = (x + dx,y + dy)


        # if observation is valid
        if valid(candidate):
            observation = candidate
        else:
            observation = state # if observation is invalid

        o_index = state_to_index[observation]

        Z[s_index,o_index] += probability








# ============================================================
# Generate random next state and observation
# ============================================================

def sample_next_state(
    current_state_index,
    action_index
):

    probabilities = T[
        action_index,
        current_state_index
    ] # gives transition prob distribution given current state and current action

    next_state_index = np.random.choice(
        NUM_STATES,
        p=probabilities
    )

    return next_state_index


def sample_observation(
    true_state_index
):

    probabilities = Z[
        true_state_index
    ]

    observation_index = np.random.choice(
        NUM_STATES,
        p=probabilities
    )

    return observation_index






# ============================================================
# Here we use a uniform prior over all non-goal states.
# ============================================================

prior = np.ones(NUM_STATES)
prior[goal_index] = 0.0
prior = (prior/prior.sum()) # normalize to a prob distribution


# Define initial belief b_0(x)
def initial_belief_update(prior,observation_index):

    likelihood = Z[:,observation_index]

    updated_belief = (prior*likelihood)

    total = updated_belief.sum()
    if total > 0:
        updated_belief = (updated_belief/total) # normalize to a prob distribution
    return updated_belief




# ============================================================
# VISUALIZE BELIEF
# ============================================================

def show_belief(
    belief,
    title="Belief"
):

    grid = np.full(
        (
            HEIGHT,
            WIDTH
        ),
        np.nan
    )

    for s_index, state in enumerate(states):

        x, y = state

        grid[
            y,
            x
        ] = belief[
            s_index
        ]

    plt.figure()

    plt.imshow(
        grid,
        origin="lower",
        vmin=0,
        vmax=max(
            0.25,
            belief.max()
        )
    )

    plt.colorbar(
        label="Probability"
    )

    for x, y in obstacles:

        plt.text(
            x,
            y,
            "X",
            ha="center",
            va="center"
        )

    gx, gy = goal

    plt.text(
        gx,
        gy,
        "G",
        ha="center",
        va="center"
    )

    plt.xticks(
        range(WIDTH)
    )

    plt.yticks(
        range(HEIGHT)
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(title)

    plt.show()