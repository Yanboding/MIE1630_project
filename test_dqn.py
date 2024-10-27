import numpy as np
from scipy.stats import poisson

from environment import SchedulingEnv
from decision_maker import DQNAgent

if __name__ == "__main__":

    treatment_patterns = np.array([[2, 1], [1, 0]]).T
    num_tracking_days, num_types = treatment_patterns.shape
    system_dynamic = [[0.5, np.array([2, 0])], [0.5, np.array([2, 2])]]
    holding_cost = np.array([10, 5])
    overtime_cost = 30
    duration = 1
    regular_capacity = 5
    discount_factor = 0.99
    '''
    demand = np.array([6, 6])
    future_appts = np.array([[0, 0], [0, 0]])
    init_state = (np.array([0,0]), demand, future_appts)
    '''
    decision_epoch = 15
    print(decision_epoch)

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
    print(prev_state, info)
  
   # Initialize the DQN agent
    state_size = prev_state.shape[0]  # Get state size from the environment
    action_size = 3  # Number of treatment types or actions
    agent = DQNAgent(state_size, action_size)

    # Train the agent for 500 episodes
    agent.train_dqn(env, episodes=500)