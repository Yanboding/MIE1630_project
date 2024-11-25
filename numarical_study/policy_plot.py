from pprint import pprint
<<<<<<< HEAD
'''
Q_{\theta}(s,a) \in R
q_network = f(s,a) s_size+a_size -> 1 
min_q = inf
best_actoin = None
for action in env.valid_action(s):
   q = q_network(s, action)
   if q < min_q:
      min_q = q
      best_actoin = a
'''
=======
import sys
import os
>>>>>>> rl_baselines_charlie
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment import SchedulingEnv
from decision_maker import OptimalAgent, DQNAgent, PolicyEvaluator
from utils import get_system_dynamic

<<<<<<< HEAD
treatment_patterns = np.array([[2,1,0], [1,0,1]]).T

arrival_mean = 2
type_proportion = [0.2, 0.8]
system_dynamic = [[0.5, np.array([2,0])], [0.5, np.array([2,1])]]
=======
treatment_patterns = np.array([[2, 1], [1, 0]]).T
arrival_mean = 3
type_proportion = [0.5, 0.5]
system_dynamic = get_system_dynamic(arrival_mean, 4, type_proportion)
>>>>>>> rl_baselines_charlie

holding_cost = np.array([10, 5])
overtime_cost = 30
duration = 1
<<<<<<< HEAD
regular_capacity = 3
discount_factor = 0.99

decision_epoch = 5
=======

evaluation_results = []
>>>>>>> rl_baselines_charlie

for i in range(0, 1):
    regular_capacity = 2
    discount_factor = 0.99
    decision_epoch = 7

<<<<<<< HEAD
optimal_agent = OptimalAgent(env, discount_factor)
init_state = np.array([0,0,0,1,2])
#optimal_agent.load('./decision_epoch_5')
optimal_agent.train(init_state, 1)
[[0.5, [2,0]],[0.5, [2,0]]]
#optimal_agent.save('.')
pprint(optimal_agent.Q)
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
=======
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
    init_state = np.array([0, 0, i, 5])
>>>>>>> rl_baselines_charlie

    optimal_agent = OptimalAgent(env, discount_factor)
    optimal_agent.train(init_state, 1)
    optimal_value = optimal_agent.get_state_value(init_state, 1)
    print(optimal_value)
    
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

    prev_state, info = rl_env.reset(init_state=init_state, t=0)

    state_size = prev_state.shape[0]
    action_size = 2
    rl_agent = DQNAgent(state_size, action_size, rl_env)

    rl_agent.train_dqn(500, decision_periods=decision_epoch, optimal_agent=optimal_value, init_state=init_state, threshold=1.025)
    rl_agent.save_model("waitlist_1_{}.h5".format(i))
    rl_agent.load_model("waitlist_1_{}.h5".format(i))

    pe = PolicyEvaluator(rl_env, rl_agent, discount_factor=0.99, V=None)
    evaluation_result = pe.evaluate(init_state, t=1)

    evaluation_results.append(evaluation_result)

print("Evaluation Results:")
pprint(evaluation_results)
