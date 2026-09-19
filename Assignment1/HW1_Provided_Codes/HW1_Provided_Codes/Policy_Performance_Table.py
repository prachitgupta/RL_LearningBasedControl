import numpy as np

import Basic_Functions as bf
import POMDP as pomdp
import MDP as mdp
import MAP as map_policy
import QMDP as qmdp_policy


def run_episode(policy_name, start_state=(0, 0)):

    true_state = bf.state_to_index[start_state]

    observation = bf.sample_observation(true_state)

    belief = bf.initial_belief_update(
        bf.prior,
        observation
    )

    total_cost = 0.0


    for t in range(bf.HORIZON):

        if true_state == bf.goal_index:

            return True, total_cost, t


        if policy_name == "MAP":

            action = map_policy.choose_action_MAP(
                belief,
                t,
                mdp.J_opt
            )

        elif policy_name == "QMDP":

            action = qmdp_policy.choose_action_QMDP(
                belief,
                t,
                mdp.J_opt
            )

        else:

            raise ValueError(
                "policy_name must be 'MAP' or 'QMDP'"
            )


        total_cost += bf.stage_cost(
            true_state,
            action
        )

        true_state = bf.sample_next_state(
            true_state,
            action
        )

        observation = bf.sample_observation(
            true_state
        )

        belief = pomdp.bayes_update(
            belief,
            action,
            observation
        )


    # Check whether the goal was reached on the final transition.
    success = (
        true_state == bf.goal_index
    )
    total_cost += bf.terminal_costs[true_state]

    return success, total_cost, bf.HORIZON


def evaluate_policy(
    policy_name,
    number_of_episodes=1000,
    start_state=(0, 0),
    random_seed=24
):

    np.random.seed(random_seed)

    successes = []
    costs = []
    times_to_goal = []


    for _ in range(number_of_episodes):

        success, total_cost, time_to_goal = run_episode(
            policy_name,
            start_state
        )

        successes.append(success)
        costs.append(total_cost)

        if success:

            times_to_goal.append(
                time_to_goal
            )


    return {
        "policy": policy_name,
        "success_rate": np.mean(successes),
        "average_cost": np.mean(costs),
        "average_time_to_goal": (
            np.mean(times_to_goal)
            if times_to_goal
            else np.nan
        )
    }


def print_table(results):

    print()
    print(
        f"{'Policy':<10}"
        f"{'Success Rate':>15}"
        f"{'Average Cost':>15}"
        f"{'Avg. Time to Goal':>20}"
    )

    print("-" * 60)


    for r in results:

        print(
            f"{r['policy']:<10}"
            f"{r['success_rate']:>15.3f}"
            f"{r['average_cost']:>15.3f}"
            f"{r['average_time_to_goal']:>20.3f}"
        )


if __name__ == "__main__":

    NUMBER_OF_EPISODES = 1000
    START_STATE = (0, 0)
    RANDOM_SEED = 24


    map_results = evaluate_policy(
        "MAP",
        NUMBER_OF_EPISODES,
        START_STATE,
        RANDOM_SEED
    )


    qmdp_results = evaluate_policy(
        "QMDP",
        NUMBER_OF_EPISODES,
        START_STATE,
        RANDOM_SEED
    )


    print_table(
        [
            map_results,
            qmdp_results
        ]
    )
