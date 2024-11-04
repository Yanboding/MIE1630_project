import numpy as np
from scipy.stats import poisson

from environment import SchedulingEnv
from decision_maker import DQNAgent, OptimalAgent
from utils import get_system_dynamic

if __name__ == "__main__":
    treatment_patterns = np.array([[2, 1, 0], [1, 0, 1], [0, 1, 2]]).T
    num_tracking_days, num_types = treatment_patterns.shape
    
    mean_arrival = 5
    maximum_arrival = 15
    type_probs = [0.2, 0.3, 0.5]
    system_dynamic = get_system_dynamic(mean_arrival, maximum_arrival, type_probs)
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
                 discount_factor=0.99,
                 problem_type='allocation'
    )
    prev_state, info = env.reset()
    print(prev_state)
  
    state_size = prev_state.shape[0] 
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    agent.train_dqn(env, episodes=500)
    # For some reasons optimal agent does not work, will investigate later. 
    optimal_agent = OptimalAgent(env, discount_factor)

    initial_state = (np.array([0, 0, 0]), np.array([1, 2, 3]))
    optimal_agent.train(initial_state, t=0)