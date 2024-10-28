import numpy as np
import argparse
import json
from scipy.stats import poisson

from environment import SchedulingEnv
from decision_maker import BAC


def get_system_dynamic(mean_arrival, maximum_arrival, type_probs):
    total_poisson = poisson(mean_arrival)
    poisson_by_type = [poisson(mean_arrival*p) for p in type_probs]
    system_dynamic = []
    def calculate_transition_dynamic(arrival_by_type):
        prob = 1
        for num, dis in zip(arrival_by_type, poisson_by_type):
            prob *= dis.pmf(num)
        prob /= total_poisson.cdf(maximum_arrival)
        return [prob, np.array(arrival_by_type)]
    def get_transition_dynamic(x, y):
        def backtracking(x, y, current_combination):
            if y == 0:
                if x == 0:
                    all_combinations.append(calculate_transition_dynamic(current_combination.copy()))
                return

            for i in range(x + 1):
                current_combination.append(i)
                backtracking(x - i, y - 1, current_combination)
                current_combination.pop()

        all_combinations = []
        backtracking(x, y, [])
        return all_combinations
    for total_arrival in range(maximum_arrival+1):
        system_dynamic += get_transition_dynamic(total_arrival, len(type_probs))
    return system_dynamic

def run(env_spec, agent_spec):




if __name__ == "__main__":
    env_spec = {
        'treatment_pattern': [[2, 1],
                              [1, 0]],
        'decision_epoch': 10,
        'system_dynamic': [[0.5, [2, 0]], [0.5, [2, 2]]],
        'holding_cost': [10, 5],
        'overtime_cost': 30,
        'duration': 1,
        'regular_capacity': 5,
        'discount_factor': 0.99,
        'problem_type': 'allocation'
    }
    agent_spec = {
        'actor': 'CNNActor',
        'critic': 'Value',
        'discount': 0.99,
        'tau': 0.97,
        'advantage_flag': False,
        'actor_args': {'feature_extractor': 'CNNFeatureExtractor'},
        'critic_args': {'fisher_num_inputs': 50, 'feature_extractor': 'CNNFeatureExtractor'},
        'actor_lr': 3e-3,
        'critic_lr': 2e-2,
        'likelihood_noise_level': 1e-4
    }
    run(env_spec, agent_spec)
