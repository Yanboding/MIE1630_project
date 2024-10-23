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

def study_1(regular_capacity, arrival_ratio, type_proportion, decision_epoch, capacity_booked):
    treatment_patterns = np.array([[1], [1]]).T
    arrival_mean = int(regular_capacity * arrival_ratio)
    system_dynamic = get_system_dynamic(arrival_mean, arrival_mean * 3, list(type_proportion))
    holding_cost = np.array([10, 5])
    overtime_cost = 30
    duration = 1
    discount_factor = 0.99

    allocation_scheduling_env = SchedulingEnv(
        treatment_pattern=treatment_patterns,
        decision_epoch=decision_epoch,
        system_dynamic=system_dynamic,
        holding_cost=holding_cost,
        overtime_cost=overtime_cost,
        duration=duration,
        regular_capacity=regular_capacity,
        discount_factor=discount_factor,
        problem_type='allocation'
    )
    optimal_agent = BAC(allocation_scheduling_env, )
    prev_state, info = allocation_scheduling_env.reset()
    total_reward = 0
    for time_step in range(100):
        action = np.array([0, 0])
        # observation, reward, terminated, truncated, info
        state, reward, done, truncated, info = allocation_scheduling_env.step(action)
        prev_state = state
        total_reward += reward
        if done or truncated:
            break
    print(total_reward)



if __name__ == "__main__":
    env_spec = {
        'episode_length': 128,
        'commission_rate': 0,
        'period': '30m',
        'coins': ['BTC', 'ETH', 'XRP', 'BNB', 'ADA'],
        'online': False,
        'features': ['close', 'high', 'low'],
        'baseAsset': 'USDT'
    }
    times = ['2018-06-01', '2020-06-01', '2021-06-01', '2022-12-31']
    start, end = 0, 1
    envs = []
    for end in range(1, len(times)):
        env_spec['start'], env_spec['end'] = times[end - 1], times[end]
        e = TradingEnvWrapper(**env_spec)
        envs.append(e)
    env = envs[0]
    state_dim, action_dim = env.observation_space.shape, env.action_space.shape
    feature_extractor = CNNFeatureExtractor(state_dim)
    agent_spec = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'actor': CNNActor,
        'critic': Value,
        'discount': 0.995,
        'tau': 0.97,
        'advantage_flag': False,
        'actor_args': {'feature_extractor': feature_extractor},
        'critic_args': {'fisher_num_inputs': 50, 'feature_extractor': feature_extractor},
        'actor_lr': 3e-3,
        'critic_lr': 2e-2,
        'likelihood_noise_level': 1e-4
    }
    agent = BAC(**agent_spec)
    batch_size = 15
    max_time_steps = 1000
    memory = ContinuousMemory(state_dim, action_dim, batch_size * max_time_steps)
    train_spec = {
        'env': env,
        'epoch_num': 15000,
        'max_time_steps': max_time_steps,
        'batch_size': batch_size,
        'replay_memory': memory,
        'svd_low_rank': agent_spec['critic_args']['fisher_num_inputs'],
        'state_coefficient': 1,
        'fisher_coefficient': 5e-5
    }
    total_rewards = agent.fit(**train_spec)
    plot_total_reward(total_rewards)

    backtest_spec = {
        'test_envs': envs,
        'agent': agent,
        'eval_times': 200,
        'metrics': ['total_rewards', 'portfolio_values']
    }
    backtest(**backtest_spec)