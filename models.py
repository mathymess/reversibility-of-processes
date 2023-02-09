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

    def forward(self, window):
        # print(window, window.shape)
        if window.shape[-1] == self.datapoint_size:
            ret = window.flatten(window.ndim - 2)
            # print(window, window.shape)
        else:
            ret = window

        ret = self.relu1(self.fc1(ret))
        ret = self.relu2(self.fc2(ret))
        ret = self.fc3(ret)
        return ret
