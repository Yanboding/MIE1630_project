from pprint import pprint

import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment import SchedulingEnv
from decision_maker import OptimalAgent
from utils import get_system_dynamic

treatment_patterns = np.array([[1], [1]]).T
num_tracking_days, num_types = treatment_patterns.shape
'''
arrival_mean = 3
type_proportion = [0.1, 0.9]
system_dynamic = get_system_dynamic(arrival_mean, arrival_mean*3, list(type_proportion))
'''
system_dynamic = [[1, np.array([0, 0])]]
print(system_dynamic)
holding_cost = np.array([1, 1])
overtime_cost = 3
duration = 1
regular_capacity = 2
discount_factor = 0.99

future_appts = np.array([[0, 0],[0,0]])
init_state = (np.array([0]), np.array([2, 0]), future_appts)
decision_epoch = len(future_appts)

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
#optimal_agent.load('.')
for w1 in [0,2]:
    for w2 in [0,2]:
        init_state = (np.array([0]), np.array([w1, w2]))
        optimal_agent.train(init_state, 1)
        optimal_agent.save('.')
        optimal_agent.action_value_3d_plot(init_state, 1)
        print(optimal_agent.policy(init_state, 1), optimal_agent.get_state_value(init_state, 1))
        print(optimal_agent.get_action_values(init_state, 1))

