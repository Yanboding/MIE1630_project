import gpytorch
import numpy as np
import torch
from gpytorch.kernels import LinearKernel, RBFKernel
from torch import nn

from environment import SchedulingEnv
from models.mlp_feature_extractor import MLPFeatureExtractor


class Value(gpytorch.models.ExactGP, nn.Module):
    # Monte-Carlo PG estimator :- Only State-value V(s) function approximation, i.e. feature extractor + value head
    # Bayesian Quadrature PG estimator :- Both state-value V(s) and action-value Q(s,a) function approximation, i.e. feature extractor + value head + GP head
    def __init__(self, env, fisher_num_inputs, gp_likelihood=None, feature_extractor=None):
        # fisher_num_inputs is same as svd_low_rank, because of the linear approximation of the Fisher kernel through FastSVD.
        gpytorch.models.ExactGP.__init__(self, None, None, gp_likelihood)
        self.env = env
        # state_dim + 1 to include time dim
        self.state_element_num = env.state_dim + 1
        self.action_element_num = env.action_dim
        NN_num_outputs = 10
        if feature_extractor:
            self.feature_extractor = feature_extractor
        else:
            # add 1 to state_element_num
            self.feature_extractor = MLPFeatureExtractor(self.state_element_num, NN_num_outputs)

        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1, 1)
        # value_head is used for computing the state-value function approximation V(s) and subsequently GAE estimates
        self.value_head = nn.Linear(NN_num_outputs, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        grid_size = 128
        # Like value_head, the following code constructs the GP head for action-value function approximation Q(s,a)
        # Note that both V(s) and Q(s,a) share the same feature extractor for the state-values "s".
        self.mean_module = gpytorch.means.ConstantMean()

        # First NN_num_outputs indices of GP's input correspond to the state_kernel
        state_kernel_active_dims = torch.tensor(list(range(NN_num_outputs)))
        # [NN_num_outputs, GP_input.shape[1]-1] indices of GP's input correspond to the fisher_kernel
        fisher_kernel_active_dims = torch.tensor(list(range(NN_num_outputs, fisher_num_inputs + NN_num_outputs)))
        self.covar_module_2 = LinearKernel(active_dims=fisher_kernel_active_dims)
        self.covar_module_1 = gpytorch.kernels.AdditiveStructureKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.GridInterpolationKernel(
                    RBFKernel(ard_num_dims=1),
                    grid_size=grid_size,
                    num_dims=1)),
            num_dims=NN_num_outputs,
            active_dims=state_kernel_active_dims)

    def nn_forward(self, state):
        # Invokes the value_head for computing the state value function V(s), subsequently used for computing GAE estimates
        extracted_features = self.feature_extractor(state)
        state_value_estimate = self.value_head(extracted_features)
        return state_value_estimate

    def forward(self, x, state_multiplier, fisher_multiplier, only_state_kernel=False):
        # Invokes the GP head for computing the action value function Q(s,a), although Q(s,a) is never explicitly computed.
        # Instead, we implicitly compute (K + sigma^2 I)^{-1}*A^{GAE} which subsequently provides the BQ's PG estimate.
        # x is [state, v_ten]
        extracted_features = self.feature_extractor(x[:, :self.state_element_num])
        extracted_features = self.scale_to_bounds(extracted_features)

        if only_state_kernel:
            # Used for computing inverse vanilla gradient covariance (Cov^{BQ})^{-1} or natural gradient covariance (Cov^{NBQ})^{-1}
            mean_x = self.mean_module(extracted_features)
            # Implicitly computes (c_1 K_s + sigma^2 I) which can be used for efficiently computing the MVM (c_1 K_s + sigma^2 I)^{-1}*v
            covar_x = gpytorch.lazy.ConstantMulLazyTensor(
                self.covar_module_1(extracted_features), state_multiplier)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        GP_input = torch.cat([extracted_features, x[:, self.state_element_num:]],1)
        mean_x = self.mean_module(GP_input)
        # Implicitly computes (c_1 K_s + c_2 K_f + sigma^2 I) which can be used for efficiently computing the MVM (c_1 K_s + c_2 K_f + sigma^2 I)^{-1}*v
        covar_x = gpytorch.lazy.ConstantMulLazyTensor(self.covar_module_1(GP_input), state_multiplier) + \
                                gpytorch.lazy.ConstantMulLazyTensor(self.covar_module_2(GP_input), fisher_multiplier)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == '__main__':
    import gpytorch.constraints as constraints

    num_sessions, num_types = 2, 2
    env_state = np.array([2, 1, 5, 5, 2]) # state, time
    vaild_action = np.array([1, 2])
    states = torch.tensor(env_state.reshape(1, *env_state.shape), dtype=torch.float32)
    actions = torch.tensor(vaild_action.reshape(1, *vaild_action.shape), dtype=torch.float32)

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

    likelihood_noise_level = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=constraints.GreaterThan(likelihood_noise_level)).to(device)
    fisher_num_inputs = 50
    critic = Value(env, fisher_num_inputs, gp_likelihood=gp_likelihood, feature_extractor=None)
    print(critic.nn_forward(states))
