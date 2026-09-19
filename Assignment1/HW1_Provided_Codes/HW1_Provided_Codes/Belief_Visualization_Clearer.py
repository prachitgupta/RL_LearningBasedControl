import numpy as np
import matplotlib.pyplot as plt

import Basic_Functions as bf
import POMDP as pomdp


# this is to be used in homework 1!

# ============================================================
# FIXED DATA FOR THE HOMEWORK
# ============================================================

# True states x_0, ..., x_8
#
# These are included here for instructor/reference comparison.
# You may hide them from students if you want students to infer
# location only through the belief state.
true_state_sequence = [
    (0, 0),
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 1),   # commanded UP, but obstacle at (2,2), so staying is possible
    (3, 1),
    (4, 1),
    (4, 2),
    (4, 3)
]

# Initial observation y_0
initial_observation = (0, 0)

# Controls u_0, ..., u_7
action_sequence = [
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP"
]

# Observations y_1, ..., y_8
#
# A few observations are deliberately noisy:
# y_4 = (1,1) while x_4 = (2,1)
# y_5 = (3,0) while x_5 = (3,1)
# y_8 = (4,2) while x_8 = (4,3)
observation_sequence = [
    (1, 0),
    (2, 0),
    (2, 1),
    (1, 1),
    (3, 0),
    (4, 1),
    (4, 2),
    (4, 2)
]


# ============================================================
# COMPUTE BELIEF TRAJECTORY
# ============================================================

initial_observation_index = bf.state_to_index[
    initial_observation
]

belief = bf.initial_belief_update(
    bf.prior,
    initial_observation_index
)

belief_history = [
    belief.copy()
]


for action_name, observation in zip(
    action_sequence,
    observation_sequence
):

    action_index = bf.action_names.index(
        action_name
    )

    observation_index = bf.state_to_index[
        observation
    ]

    belief = pomdp.bayes_update(
        belief,
        action_index,
        observation_index
    )

    belief_history.append(
        belief.copy()
    )


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\nTime | True state | Observation | Belief at true state")
print("-------------------------------------------------------")

all_observations = [
    initial_observation
] + observation_sequence

for t, belief in enumerate(
    belief_history
):

    true_state = true_state_sequence[t]

    true_state_index = bf.state_to_index[
        true_state
    ]

    probability_at_true_state = belief[
        true_state_index
    ]

    print(
        f"{t:>4} | "
        f"{str(true_state):>10} | "
        f"{str(all_observations[t]):>11} | "
        f"{probability_at_true_state:.4f}"
    )


# ============================================================
# VISUALIZE BELIEF + TRUE STATE
# ============================================================

def show_belief_trajectory(
    belief_history,
    true_state_sequence,
    observations
):

    # One common probability scale for every plot.
    common_vmax = max(
        belief.max()
        for belief in belief_history
    )

    number_of_plots = len(
        belief_history
    )

    ncols = 3

    nrows = int(
        np.ceil(
            number_of_plots / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4 * nrows)
    )

    axes = np.array(
        axes
    ).reshape(-1)


    for t, belief in enumerate(
        belief_history
    ):

        ax = axes[t]

        grid = np.full(
            (bf.HEIGHT, bf.WIDTH),
            np.nan
        )

        for state_index, state in enumerate(
            bf.states
        ):

            x, y = state

            grid[y, x] = belief[
                state_index
            ]


        image = ax.imshow(
            grid,
            origin="lower",
            vmin=0,
            vmax=common_vmax,
            cmap="YlOrRd"
        )


        # --------------------------------------------
        # Write belief probability in each valid cell
        # --------------------------------------------

        for state_index, state in enumerate(
            bf.states
        ):

            x, y = state

            value = belief[
                state_index
            ]

            # Use white text only on high-probability/dark cells.
            text_color = (
                "white"
                if value > 0.55 * common_vmax
                else "black"
            )

            ax.text(
                x,
                y,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                zorder=5
            )


        # --------------------------------------------
        # Mark obstacles
        # --------------------------------------------

        for x, y in bf.obstacles:

            ax.text(
                x,
                y,
                "X",
                ha="center",
                va="center",
                fontsize=12
            )


        # --------------------------------------------
        # Mark goal
        # --------------------------------------------

        gx, gy = bf.goal

        ax.text(
            gx,
            gy,
            "G",
            ha="center",
            va="bottom",
            fontsize=10
        )


        # --------------------------------------------
        # Mark the TRUE state
        # --------------------------------------------

        true_x, true_y = true_state_sequence[
            t
        ]

        # Small marker near the upper-left corner of the true-state cell,
        # rather than on top of the probability value.
        ax.scatter(
            true_x - 0.32,
            true_y + 0.32,
            marker="*",
            s=90,
            color="red",
            edgecolors="white",
            linewidths=0.7,
            zorder=7,
            label="True state"
        )


        # --------------------------------------------
        # Mark the OBSERVATION
        # --------------------------------------------

        obs_x, obs_y = observations[
            t
        ]

        # Small marker near the lower-right corner of the observed cell.
        ax.scatter(
            obs_x + 0.32,
            obs_y - 0.32,
            marker="o",
            s=55,
            facecolors="none",
            edgecolors="blue",
            linewidths=1.8,
            zorder=7,
            label="Observation"
        )


        ax.set_xticks(
            range(bf.WIDTH)
        )

        ax.set_yticks(
            range(bf.HEIGHT)
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_title(
            f"$b_{t}$: true={true_state_sequence[t]}, "
            f"obs={observations[t]}"
        )


    # Hide unused subplot positions.
    for k in range(
        number_of_plots,
        len(axes)
    ):

        axes[k].axis("off")


    # Shared color bar
    fig.colorbar(
        image,
        ax=axes[:number_of_plots].tolist(),
        label="Belief probability",
        shrink=0.8
    )

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2
    )

    plt.suptitle(
        "Belief-State Evolution Compared with True State",
        y=0.98
    )

    plt.show()


all_observations = [
    initial_observation
] + observation_sequence

show_belief_trajectory(
    belief_history,
    true_state_sequence,
    all_observations
)
