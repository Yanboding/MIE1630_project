from pprint import pprint
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from environment import SchedulingEnv
from decision_maker import OptimalAgent, DQNAgent
from utils import get_system_dynamic

treatment_patterns = np.array([[2,1], [1,0]]).T

arrival_mean = 3
type_proportion = [0.5, 0.5]
system_dynamic = get_system_dynamic(3, 4, [0.5, 0.5])

holding_cost = np.array([10, 5])
overtime_cost = 30
duration = 1
regular_capacity = 2
discount_factor = 0.99

decision_epoch = 5

env = SchedulingEnv(
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

optimal_agent = OptimalAgent(env, discount_factor)
init_state = (np.array([0,0]), np.array([0, 0]))
optimal_agent.train(init_state, 1)
print(optimal_agent.policy(init_state, 1), optimal_agent.get_state_value(init_state, 1))

rl_env = SchedulingEnv(
                 treatment_pattern=treatment_patterns,
                 decision_epoch=decision_epoch,
                 system_dynamic=system_dynamic,
                 holding_cost=holding_cost,
                 overtime_cost=overtime_cost,
                 duration=duration,
                 regular_capacity=regular_capacity,
                 discount_factor=0.99,
                 problem_type='allocation'
    )

prev_state, info = rl_env.reset()
print(prev_state)

state_size = prev_state.shape[0] 
action_size = 2
rl_agent = DQNAgent(state_size, action_size, rl_env)

rl_agent.train_dqn(1000, 10)
