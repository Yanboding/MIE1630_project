import os

import numpy as np
import pandas as pd
from decision_maker.policy_evaluator import PolicyEvaluator
from environment import SchedulingEnv
from decision_maker import BAC, OptimalAgent
from models import Value, MLPFeatureExtractor
from models.mlp_policy_disc import DiscretePolicy
from utils import ContinuousMemory, get_system_dynamic
import matplotlib.pyplot as plt

def interoperate_agent_spec(agent_spec, env):
    if agent_spec['actor'] == 'MLPDiscrete':
        actor = DiscretePolicy
    else:
        raise ValueError('Not implement')
    if agent_spec['critic'] == 'BACCritic':
        critic = Value
    else:
        raise ValueError('Not implement')
    if agent_spec['feature_extractor'] == 'MLPFeatureExtractor':
        feature_extractor = MLPFeatureExtractor
    else:
        raise ValueError('Not implement')
    discount = agent_spec['discount']
    tau = agent_spec['tau']
    advantage_flag = agent_spec['advantage_flag']
    actor_args = {}
    critic_args = {'fisher_num_inputs': agent_spec['fisher_num_inputs']}
    actor_lr = agent_spec['actor_lr']
    critic_lr = agent_spec['critic_lr']
    likelihood_noise_level = agent_spec['likelihood_noise_level']
    agent = BAC(env,
                actor=actor,
                critic=critic,
                discount=discount,
                tau=tau,
                advantage_flag=advantage_flag,
                actor_args=actor_args,
                critic_args=critic_args,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                likelihood_noise_level=likelihood_noise_level)
    return agent

def create_env(env_spec):
    treatment_pattern = np.array(env_spec['treatment_pattern']).T
    decision_epoch = env_spec['decision_epoch']
    system_dynamic = [[prob, np.array(delta)] for prob, delta in env_spec['system_dynamic']]
    holding_cost = np.array(env_spec['holding_cost'])
    overtime_cost = env_spec['overtime_cost']
    duration = env_spec['duration']
    regular_capacity = env_spec['regular_capacity']
    discount_factor = env_spec['discount_factor']
    future_first_appts = env_spec.get('future_first_appts', None)
    problem_type = env_spec['problem_type']
    env = SchedulingEnv(
        treatment_pattern=treatment_pattern,
        decision_epoch=decision_epoch,
        system_dynamic=system_dynamic,
        holding_cost=holding_cost,
        overtime_cost=overtime_cost,
        duration=duration,
        regular_capacity=regular_capacity,
        discount_factor=discount_factor,
        future_first_appts=future_first_appts,
        problem_type=problem_type
    )
    return env

def run(env_spec, agent_spec):
    env = create_env(env_spec)
    #optimal_agent = OptimalAgent(env, discount_factor)
    #init_state = (np.array([0, 0]), np.array([0, 0]))
    # optimal_agent.load('./decision_epoch_5')
    #optimal_agent.train(init_state, 1)
    # optimal_agent.save('.')
    #print(optimal_agent.policy(init_state, 1), optimal_agent.get_state_value(init_state, 1))
    agent = interoperate_agent_spec(agent_spec, env)
    batch_size = 15
    max_time_steps = 1000
    memory = ContinuousMemory(env.state_dim, env.action_dim, batch_size * max_time_steps)
    train_spec = {
        'epoch_num': 500,
        'max_time_steps': max_time_steps,
        'batch_size': batch_size,
        'replay_memory': memory,
        'svd_low_rank': agent_spec['fisher_num_inputs'],
        'state_coefficient': 1,
        'fisher_coefficient': 5e-5
    }
    total_rewards = agent.train(**train_spec)
    return total_rewards, env, agent

def set_fontsize(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

def opt_plot(df, xlable, save_file):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    x_vals = df['decision_epoch']
    ax.plot(df['decision_epoch'], df['optimal_value'], label='Optimal Policy', marker='o')
    ax.plot(df['decision_epoch'], df['bac_value'], label='BAC Algorithm', marker='o')
    #ax.plot(df['decision_epoch'], df['allocation_value'], label='Allocation Scheduling', marker='o')
    for l, (upper_txt, opt_txt) in enumerate(zip(df['bac_value'], df['optimal_value'])):
        ax.text(x_vals[l], df['bac_value'][l], str(round(upper_txt,3)), ha='center', va='bottom', fontsize=20)
        ax.text(x_vals[l], df['optimal_value'][l], str(round(opt_txt,3)), ha='center', va='bottom', fontsize=20)
        #ax.text(x_vals[l], df['allocation_value'][l], str(round(lower_txt,3)), ha='center', va='bottom', fontsize=20)
    set_fontsize(ax, 20)
    # To handle multiple lines with the same label, we need to manually create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = sorted(list(set(labels)))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_vals, rotation=0, fontsize=20)
    ax.set_xlabel(xlable, fontsize=20)
    ax.set_ylabel('Value Function', fontsize=20)
    # Create legend
    ax.legend(unique_handles, unique_labels, fontsize=20)
    fig.tight_layout()
    plt.savefig(save_file)
    plt.show()

def plot_train_cost(cost, optimal_value, save_file):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # Plotting the rewards
    ax.plot(cost, marker='o', linestyle='-', label='Estimate Total Cost')
    ax.axhline(y=optimal_value, color='r', linestyle='--', label='Optimal Value')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    set_fontsize(ax, 20)
    plt.savefig(save_file)
    plt.show()


if __name__ == "__main__":
    env_spec = {
        'treatment_pattern': [[2, 1],
                              [1, 0]],
        'decision_epoch': 7,
        'system_dynamic': get_system_dynamic(3, 4, [0.5, 0.5]),
        'holding_cost': [10, 5],
        'overtime_cost': 30,
        'duration': 1,
        'regular_capacity': 2,
        'discount_factor': 0.99,
        'problem_type': 'allocation'
    }
    agent_spec = {
        'actor': 'MLPDiscrete',
        'critic': 'BACCritic',
        'discount': 0.99,
        'tau': 0.97,
        'advantage_flag': True,
        'feature_extractor': 'MLPFeatureExtractor',
        'fisher_num_inputs': 20,
        'actor_lr': 3e-3,
        'critic_lr': 2e-2,
        'likelihood_noise_level': 1e-4
    }
    #total_rewards, env, agent = run(env_spec, agent_spec)

    env = create_env(env_spec)
    '''
    optimal_agent = OptimalAgent(env=env, discount_factor=0.99)
    init_state = np.array([0, 0, 0, 0])
    optimal_agent.train(init_state, 1)
    optimal_value = optimal_agent.get_state_value(init_state, 1)
    print(optimal_value)

    agent = interoperate_agent_spec(agent_spec, env)
    '''
    batch_size = 15
    max_time_steps = 1000
    memory = ContinuousMemory(env.state_dim, env.action_dim, batch_size * max_time_steps)
    train_spec = {
        'init_state': [0, 0, 5, 5],
        'init_t': 0,
        'epoch_num': 500,
        'max_time_steps': max_time_steps,
        'batch_size': batch_size,
        'replay_memory': memory,
        'svd_low_rank': agent_spec['fisher_num_inputs'],
        'state_coefficient': 1,
        'fisher_coefficient': 5e-5,
    }
    #costs = np.array(agent.train(**train_spec))*(-1)
    #print(cost)
    #plot_train_cost(costs, optimal_value, save_file="train_cost.png")
    #print(evaluator)

    '''
    we want to plot value function for bac
     period_to_go
    '''
    '''
    df = []
    for period_to_go in range(5, 11):
        env_spec['decision_epoch'] = period_to_go
        env = create_env(env_spec)
        agent = interoperate_agent_spec(agent_spec, env)
        train_spec['init_state'] = [0, 0, 5, 5]
        agent.train(**train_spec)
        evaluator = PolicyEvaluator(env, agent, 0.99)
        init_state = np.array(train_spec['init_state'])
        bac_value = evaluator.evaluate(init_state, 1)
        optimal_agent = OptimalAgent(env=env, discount_factor=0.99)
        optimal_agent.train(init_state, 1)
        optimal_value = optimal_agent.get_state_value(init_state, 1)
        df.append([period_to_go, bac_value, optimal_value])
        print([period_to_go, bac_value, optimal_value])
    df = pd.DataFrame(df, columns=['decision_epoch', 'bac_value', 'optimal_value'])
    data_path = os.path.join('.', 'study_2.csv')
    df.to_csv(data_path, mode='a', index=False, header=False)
    opt_plot(df, save_file="period_to_go_proposal.png")
    '''
    '''
    df = []
    for type_1 in range(11):
        env = create_env(env_spec)
        agent = interoperate_agent_spec(agent_spec, env)
        train_spec['init_state'] = [0, 0, type_1, 5]
        agent.train(**train_spec)
        evaluator = PolicyEvaluator(env, agent, 0.99)
        init_state = np.array(train_spec['init_state'])
        bac_value = evaluator.evaluate(init_state, 1)
        optimal_agent = OptimalAgent(env=env, discount_factor=0.99)
        optimal_agent.train(init_state, 1)
        optimal_value = optimal_agent.get_state_value(init_state, 1)
        df.append([type_1, bac_value, optimal_value])
        print([type_1, bac_value, optimal_value])
        data_path = os.path.join('.', 'type_1.csv')
        pd.DataFrame([[type_1, bac_value, optimal_value]], columns=['type_1_num', 'bac_value', 'optimal_value']).to_csv(data_path, mode='a', index=False, header=False)
    df = pd.DataFrame(df, columns=['type_1_num', 'bac_value', 'optimal_value'])
    opt_plot(df, xlable='Type 1 Number', save_file="type_1_proposal.png")
    '''
    '''
    df = []
    for type_2 in range(11):
        env = create_env(env_spec)
        agent = interoperate_agent_spec(agent_spec, env)
        train_spec['init_state'] = [0, 0, 5, type_2]
        agent.train(**train_spec)
        evaluator = PolicyEvaluator(env, agent, 0.99)
        init_state = np.array(train_spec['init_state'])
        bac_value = evaluator.evaluate(init_state, 1)
        optimal_agent = OptimalAgent(env=env, discount_factor=0.99)
        optimal_agent.train(init_state, 1)
        optimal_value = optimal_agent.get_state_value(init_state, 1)
        df.append([type_2, bac_value, optimal_value])
        print([type_2, bac_value, optimal_value])
        data_path = os.path.join('.', 'type_2.csv')
        pd.DataFrame([[type_2, bac_value, optimal_value]], columns=['type_2_num', 'bac_value', 'optimal_value']).to_csv(
            data_path, mode='a', index=False, header=False)
    df = pd.DataFrame(df, columns=['decision_epoch', 'bac_value', 'optimal_value'])
    opt_plot(df, xlable='Type 2 Number', save_file="type_2_proposal.png")
    '''
    df = []
    for capacity in range(2,5):
        env_spec['regular_capacity'] = capacity
        env = create_env(env_spec)
        agent = interoperate_agent_spec(agent_spec, env)
        train_spec['init_state'] = [0, 0, 5, 5]
        agent.train(**train_spec)
        evaluator = PolicyEvaluator(env, agent, 0.99)
        init_state = np.array(train_spec['init_state'])
        bac_value = evaluator.evaluate(init_state, 1)
        optimal_agent = OptimalAgent(env=env, discount_factor=0.99)
        optimal_agent.train(init_state, 1)
        optimal_value = optimal_agent.get_state_value(init_state, 1)
        df.append([capacity, bac_value, optimal_value])
        print([capacity, bac_value, optimal_value])
        data_path = os.path.join('.', 'type_2.csv')
        pd.DataFrame([[capacity, bac_value, optimal_value]], columns=['capacity', 'bac_value', 'optimal_value']).to_csv(
            data_path, mode='a', index=False, header=False)
    df = pd.DataFrame(df, columns=['decision_epoch', 'bac_value', 'optimal_value'])
    opt_plot(df, xlable='Regular Capacity', save_file="capacity_proposal.png")
