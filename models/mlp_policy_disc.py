import numpy as np
import torch.nn as nn
import torch
from itertools import product
from utils import compute_flat_grad
from environment import SchedulingEnv

class DiscretePolicy(nn.Module):
    def __init__(self, env, hidden_size=(128, 128), activation='tanh', feature_extractor=None):
        super().__init__()
        self.env = env
        self.max_allocations = env.maximum_allocation
        self.action_space = torch.tensor(list(product(*[range(max_alloc + 1) for max_alloc in env.maximum_allocation])))
        self.max_action_space_size = len(self.action_space)
        #self.feature_extractor = feature_extractor
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = env.state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, self.max_action_space_size)

    def get_action_mask(self, states):
        waitlist = states[:, self.env.num_sessions:]
        mask = torch.zeros((waitlist.shape[0], self.max_action_space_size))
        for i, w in enumerate(waitlist):
            for j, action in enumerate(self.action_space):
                # Valid if allocation <= patients
                if torch.all(action <= w).item():
                    mask[i, j] = 1
        return mask

    def get_action_index(self, actions):
        action_idxs  = []
        for i, a in enumerate(actions):
            for j, action in enumerate(self.action_space):
                if torch.all(action == a).item():
                    action_idxs.append(j)
        return torch.tensor(action_idxs)

    def forward(self, states):
        action_masks = self.get_action_mask(states)
        #states = self.feature_extractor(states)
        for affine in self.affine_layers:
            states = self.activation(affine(states))
        # https://ai.stackexchange.com/questions/2980/how-should-i-handle-invalid-actions-when-using-reinforce#:~:text=An%20experimental%20paper,experience%20memory%20too.
        # a trick to set the probability of invalid actions to 0
        action_prob = torch.softmax(self.action_head(states) + torch.log(action_masks + 1e-9), dim=1)
        return action_prob

    def select_action(self, states, train=True):
        action_prob = self.forward(states)
        if train:
            action = action_prob.multinomial(1)
        else:
            action = torch.argmax(action_prob, dim=1, keepdim=True)
        return self.action_space[action]

    def get_log_prob(self, states, actions):
        action_prob = self.forward(states)
        action_indexes = self.get_action_index(actions)
        return torch.log(action_prob.gather(1, action_indexes.long().unsqueeze(1)))

    def get_fim(self, states):
        action_prob = self.forward(states)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}

    def Fvp_fim(self, v, states, damping=1e-1):
        M, mu, info = self.get_fim(states)
        mu = mu.view(-1)
        filter_input_ids = set()

        t = torch.ones(mu.size(), requires_grad=True, device=mu.device)
        # see https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, self.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * v).sum()
        Jv = torch.autograd.grad(Jtv, t)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, self.parameters(), filter_input_ids=filter_input_ids).detach()
        JTMJv /= states.shape[0]
        return JTMJv + v * damping

if __name__ == '__main__':
    num_sessions, num_types = 2, 2
    env_state = np.array([2, 1, 5, 5])
    vaild_action = np.array([1, 2])
    states = torch.tensor([[2, 1, 5, 5],
                                 [2, 1, 4, 4]], dtype=torch.float32)
    actions = torch.tensor([[1, 2], [2, 2]], dtype=torch.float32)

    env = SchedulingEnv(treatment_pattern=np.array([[2, 1], [1, 0]]).T,
                        decision_epoch=5,
                        system_dynamic=[[0.5, np.array([2, 0])], [0.5, np.array([2, 2])]],
                        holding_cost=np.array([10, 5]),
                        overtime_cost=30,
                        duration=1,
                        regular_capacity=5,
                        discount_factor=0.99,
                        future_first_appts=None,
                        problem_type='allocation')
    actor = DiscretePolicy(env, hidden_size=(128, 128), activation='tanh')
    print(actor.get_log_prob(states, actions))
