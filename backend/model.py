import torch
import torch.nn as nn
import torch.optim as optim

class MysteryNet(nn.Module):
    def __init__(self, input_dim=205, hidden_dim=512, output_dim=5):
        super(MysteryNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x) # Logits
        return x

    def get_weights_structure(self):
        """
        Returns a simplified structure of weights for visualization.
        For 512 neurons, sending all weights is too heavy (205*512 ~ 100k floats).
        We might need to send a summary or just the architecture for now, 
        or a subset if the user really wants to see 'weights inside'.
        
        However, to visualize the 'network', we need nodes and edges.
        """
        return {
            "layer1": {
                "weight": self.layer1.weight.data.tolist(),
                "bias": self.layer1.bias.data.tolist()
            },
            "layer2": {
                "weight": self.layer2.weight.data.tolist(),
                "bias": self.layer2.bias.data.tolist()
            },
            "output": {
                "weight": self.output_layer.weight.data.tolist(),
                "bias": self.output_layer.bias.data.tolist()
            }
        }
