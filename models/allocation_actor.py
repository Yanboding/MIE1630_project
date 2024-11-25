import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product


class AllocationActor(nn.Module):
    def __init__(self, state_dim, max_allocations):
        """
        Args:
        - state_dim: Number of classes (size of the state vector).
        - max_allocations: List of maximum possible patients in each class.
        """
        super(AllocationActor, self).__init__()
        self.state_dim = state_dim
        self.max_allocations = max_allocations
        self.max_action_space_size = self.compute_max_action_space_size()

        # Network layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, self.max_action_space_size)

    def compute_max_action_space_size(self):
        # Compute the total number of possible allocations for the largest state
        return torch.prod(torch.tensor([max_alloc + 1 for max_alloc in self.max_allocations])).item()

    def forward(self, state):
        """
        Forward pass through the network.
        Args:
        - state: Tensor of shape (batch_size, state_dim).
        Returns:
        - action_scores: Tensor of shape (batch_size, max_action_space_size).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_scores = self.fc_out(x)  # Raw logits for all possible actions
        return action_scores


def generate_action_space(max_allocations):
    """
    Generate all possible allocations based on max allocations.
    Args:
    - max_allocations: List of maximum possible patients in each class.
    Returns:
    - actions: List of all possible allocations (tuples).
    """
    return list(product(*[range(max_alloc + 1) for max_alloc in max_allocations]))


def apply_mask(action_scores, state, action_space):
    """
    Apply a mask to filter invalid actions based on the current state.
    Args:
    - action_scores: Raw logits for all possible actions.
    - state: Current state of the waitlist (batch_size, state_dim).
    - action_space: List of all possible actions (tuples).
    Returns:
    - masked_scores: Logits with invalid actions masked.
    """
    batch_size = state.size(0)
    max_action_space_size = len(action_space)
    mask = torch.zeros((batch_size, max_action_space_size))

    for i, s in enumerate(state):
        for j, action in enumerate(action_space):
            if all(action[k] <= s[k] for k in range(len(s))):  # Valid if allocation <= patients
                mask[i, j] = 1

    masked_scores = action_scores + (mask - 1) * 1e9  # Mask invalid actions
    return masked_scores


# Example Usage
if __name__ == "__main__":
    # State dimensions and maximum allocations
    state_dim = 2
    max_allocations = [5, 3]  # Max patients in each class

    # Initialize the actor network
    actor = AllocationActor(state_dim, max_allocations)

    # Generate the action space
    action_space = generate_action_space(max_allocations)

    # Dummy state input (batch_size=2)
    state = torch.tensor([[5, 3], [2, 1]], dtype=torch.float32)

    # Get raw action scores
    action_scores = actor(state)

    # Apply mask
    masked_scores = apply_mask(action_scores, state, action_space)

    # Select actions using argmax
    selected_actions = torch.argmax(masked_scores, dim=1)
    print("Selected Actions (Indices):", selected_actions)
    print("Selected Actions:", [action_space[idx] for idx in selected_actions])