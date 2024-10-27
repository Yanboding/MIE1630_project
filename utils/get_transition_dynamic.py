from scipy.stats import poisson
import numpy as np

def get_system_dynamic(mean_arrival, maximum_arrival, type_probs):
    total_poisson = poisson(mean_arrival)
    poisson_by_type = [poisson(mean_arrival * p) for p in type_probs]
    system_dynamic = []

    def calculate_transition_dynamic(arrival_by_type):
        prob = 1
        for num, dis in zip(arrival_by_type, poisson_by_type):
            prob *= dis.pmf(num) 
        prob /= total_poisson.cdf(maximum_arrival)
        return [prob, np.array(arrival_by_type)]  

    def get_transition_dynamic(total_arrival, num_types):
        def backtracking(total_arrival, num_types, current_combination):
            if num_types == 0:
                if total_arrival == 0:
                    all_combinations.append(calculate_transition_dynamic(current_combination.copy()))
                return

            for i in range(total_arrival + 1):
                current_combination.append(i)
                backtracking(total_arrival - i, num_types - 1, current_combination)
                current_combination.pop()

        all_combinations = []
        backtracking(total_arrival, num_types, [])
        return all_combinations

    for total_arrival in range(maximum_arrival + 1):
        system_dynamic += get_transition_dynamic(total_arrival, len(type_probs))

    return system_dynamic