from pprint import pprint

import numpy as np

from environment import SchedulingEnv
from decision_maker import OptimalAgent
from utils import get_system_dynamic

treatment_patterns = np.array([[2,1,0], [1,0,1], [0,1,2]]).T

arrival_mean = 5
type_proportion = [0.2, 0.3, 0.5]
system_dynamic = get_system_dynamic(arrival_mean, arrival_mean*3, list(type_proportion))

holding_cost = np.array([10, 5, 8])
overtime_cost = 30
duration = 1
regular_capacity = 5
discount_factor = 0.99

decision_epoch = 15

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
init_state = (np.array([0,0,0]), np.array([1, 2, 3]))
optimal_agent.train(init_state, 1)
print(optimal_agent.policy(init_state, 1), optimal_agent.get_state_value(init_state, 1))
'''
#optimal_agent.load('.')
for w1 in range(10):
    for w2 in range(10):
        init_state = (np.array([0,0,0]), np.array([1, 2,3]))
        optimal_agent.train(init_state, 1)
        optimal_agent.save('.')
        optimal_agent.action_value_3d_plot(init_state, 1)
        print(optimal_agent.policy(init_state, 1), optimal_agent.get_state_value(init_state, 1))
        print(optimal_agent.get_action_values(init_state, 1))
'''

