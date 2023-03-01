import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, window_len: int, datapoint_size: int):
        super().__init__()

        self.datapoint_size: int = datapoint_size

        self.hidden_layer_size1: int = 100
        self.hidden_layer_size2: int = 100

        self.fc1 = nn.Linear(in_features=window_len * datapoint_size,
                             out_features=self.hidden_layer_size1)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=self.hidden_layer_size1,
                             out_features=self.hidden_layer_size2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=self.hidden_layer_size2,
                             out_features=datapoint_size)

    def forward(self, windows: torch.tensor):
        # Flatten the last dimension of 'windows', i.e. remove []'s
        # around each datapoint in the [batch of [windows of [datapoints]]].
        ret = windows.flatten(start_dim=-2, end_dim=-1)

        ret = self.relu1(self.fc1(ret))
        ret = self.relu2(self.fc2(ret))
        ret = self.fc3(ret)
        return ret
