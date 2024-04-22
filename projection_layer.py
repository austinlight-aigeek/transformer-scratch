import torch
import torch.nn as nn


# Building Linear Layer
class ProjectionLayer(nn.Module):

    # Model dimension and the size of the output vocabulary
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        # Linear layer for projecting the feature space of 'd_model' to the output space of 'vocab_size'
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Applying the log Softmax function to the output
        return torch.log_softmax(self.proj(x), dim = -1)