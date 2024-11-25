from pprint import pprint

import numpy as np

from environment import SchedulingEnv
from decision_maker import OptimalAgent
from utils import get_system_dynamic

treatment_patterns = np.array([[2, 1], [1, 0]]).T
num_tracking_days, num_types = treatment_patterns.shape
arrival_mean = 3
type_proportion = [0.1, 0.9]
system_dynamic = get_system_dynamic(arrival_mean, arrival_mean*3, list(type_proportion))
print(system_dynamic)
holding_cost = np.array([10, 5])
overtime_cost = 30
duration = 1
regular_capacity = 3
discount_factor = 0.99

future_appts = np.array([[0, 0],[0,0],[0, 0],[0,0],[0, 0]])
init_state = (np.array([0]), np.array([5, 5]), future_appts)
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
optimal_agent.load('.')
for w1 in range(11):
    for w2 in range(11):
        init_state = (np.array([0]), np.array([w1, w2]), future_appts)
        optimal_agent.train(init_state, 1)
        optimal_agent.save('.')
optimal_agent.state_value_3d_plot(1)