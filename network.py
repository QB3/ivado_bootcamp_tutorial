import torch.nn as nn


class FCNet(nn.Module):
    def __init__(self, in_size, out_size, device, hidden_size=40, num_hidden_layers=1):
        super(FCNet, self).__init__()

        layers = []
        current_size = in_size

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size

        self.shared_net = nn.Sequential(*layers)

        # Policy head (logits)
        self.policy_head = nn.Linear(current_size, out_size)

        # Value head (single value output)
        self.value_head = nn.Linear(current_size, 1)

        self.to(device)

    def forward(self, x):
        shared_features = self.shared_net(x)
        logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return value, logits
