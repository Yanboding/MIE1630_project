from utils import iter_to_tuple

class PolicyEvaluator:

    def __init__(self, env, agent, discount_factor, V=None):
        self.env = env
        self.agent = agent
        self.discount_factor = discount_factor
        if V is None:
            self.V = {}

    def evaluate(self, state, t, use_dqn=True):
        state_tuple = iter_to_tuple(state)
        if use_dqn == True:
            action = self.agent.action(state, t, train=False)
        else:
            action = self.agent.policy(state, t)
        q = 0
        for prob, next_state, cost, done in self.env.transition_dynamic(state, action, t):
            next_state_tuple = iter_to_tuple(next_state)
            if (next_state_tuple, t + 1) in self.V:
                next_state_val = self.V[(next_state_tuple, t + 1)]
            elif done:
                next_state_val = 0
            else:
                next_state_val = self.evaluate(next_state, t + 1)
            q += prob * (cost + self.discount_factor * next_state_val)
        self.V[(state_tuple, t)] = q
        return self.V[(state_tuple, t)]

