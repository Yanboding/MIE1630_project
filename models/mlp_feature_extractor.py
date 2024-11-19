import torch
from torch import nn

class MLPFeatureExtractor(nn.Module):
    # Common feature extractor shared between the state-value V(s) and action-value Q(s,a) function approximations
    def __init__(self, num_inputs, num_outputs):
        super(MLPFeatureExtractor, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 48)
        self.affine3 = nn.Linear(48, num_outputs)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        x = torch.tanh(self.affine3(x))
        return x
