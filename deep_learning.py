import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, sequence_num, feature_num, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_num * feature_num, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input data
        Return:
            torch.Tensor: Prediction of closing values
        """
        return self.layers(x)
