import numpy as np

from environment import SchedulingEnv
from decision_maker import BAC
from models import Value, MLPFeatureExtractor
from models.mlp_policy_disc import DiscretePolicy
from utils import ContinuousMemory

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
    actor_args = {'feature_extractor': feature_extractor}
    critic_args = {'fisher_num_inputs': agent_spec['fisher_num_inputs'], 'feature_extractor': feature_extractor}
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

def run(env_spec, agent_spec):
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
    agent = interoperate_agent_spec(agent_spec, env)
    batch_size = 15
    max_time_steps = 1000
    memory = ContinuousMemory(env.state_dim, env.action_dim, batch_size * max_time_steps)
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
    total_rewards = agent.train(**train_spec)




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
        'actor': 'MLPDiscrete',
        'critic': 'BACCritic',
        'discount': 0.99,
        'tau': 0.97,
        'advantage_flag': False,
        'feature_extractor': 'MLPFeatureExtractor',
        'fisher_num_inputs': 50,
        'actor_lr': 3e-3,
        'critic_lr': 2e-2,
        'likelihood_noise_level': 1e-4
    }
    run(env_spec, agent_spec)
