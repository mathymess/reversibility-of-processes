import torch
import torch.nn as nn


def get_number_of_parameters_in_model(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ThreeFullyConnectedLayers(nn.Module):
    def __init__(self, window_len: int, datapoint_size: int, target_len: int = 1,
                 hidden_layer1_size: int = 100, hidden_layer2_size: int = 100):
        super().__init__()

        self.window_len: int = window_len
        self.datapoint_size: int = datapoint_size
        self.target_len: int = target_len

        self.hidden_layer1_size = hidden_layer1_size
        self.hidden_layer2_size = hidden_layer2_size

        self._init_layers()

    def _init_layers(self):
        self.fc1 = nn.Linear(in_features=self.window_len * self.datapoint_size,
                             out_features=self.hidden_layer1_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=self.hidden_layer1_size,
                             out_features=self.hidden_layer2_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=self.hidden_layer2_size,
                             out_features=self.target_len * self.datapoint_size)

    def forward(self, windows: torch.tensor) -> torch.tensor:
        # Flatten the last dimension of 'windows', i.e. remove []'s
        # around each datapoint in the [batch of [windows of [datapoints]]].
        ret = windows.flatten(start_dim=-2, end_dim=-1)

        ret = self.relu1(self.fc1(ret))
        ret = self.relu2(self.fc2(ret))
        ret = self.fc3(ret)

        # Unflatten the last dimension, return back the []'s around each datapoint.
        ret = ret.unflatten(dim=-1, sizes=(self.target_len, self.datapoint_size))

        return ret


def test_model_output_dimensions() -> None:
    model = ThreeFullyConnectedLayers(window_len=2, datapoint_size=3, target_len=2)

    window = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    target = model(window)
    assert target.ndim == 2 and target.shape == (2, 3)

    batch = torch.tensor([[[1., 2., 3.], [4., 5., 6.]],
                          [[7., 8., 9.], [0., 3., 7.]]])
    targets = model(batch)
    assert targets.ndim == 3 and targets.shape == (2, 2, 3)

    print("test_model_output_dimensions passed successfully!")


if __name__ == "__main__":
    test_model_output_dimensions()
