import torch
import torch.nn as nn
from ..datasets.pa_dataset import PADataset


class Dataset(PADataset):
    """
    The dataset configuration for RVTDNN.

    Parameters:
    - root_dir: The root directory of the dataset file in .mat format
    - memory: The memory depth of the RVTDNN input samples
    - train: If True, return the training set, otherwise return the validation set
    - inverse: If True, return the inverse modeling (DPD) dataset

    Note:
    Data array pattern (take memory = 2 as example)
    - Input
        i-2 i-1 i0 i1
        i-1 i0  i1 i2
        i0  i1  i2 i3
        q-2 q-1 q1
        q-1 q0  q2
        q0  q1  q3 ...
    - Target:
        i0  i1  i2 i3
        q0  i1  q2 q3 ...
    """

    def __init__(
        self,
        root_dir=None,
        pa_input=0,
        pa_output=0,
        train_ratio=0.8,
        memory=3,
        train=True,
        inverse=False,
    ):
        super().__init__(root_dir, pa_input, pa_output, train_ratio, train, inverse)
        self.train = train
        self.train_ratio = train_ratio
        self.memory = memory

    def __len__(self):
        if self.train:
            return int(super().__len__() * self.train_ratio)
        else:
            return super().__len__() - int(super().__len__() * self.train_ratio)

    def __getitem__(self, index):
        inputs = torch.zeros((self.memory + 1) * 2)
        target = torch.zeros(2)
        index = super().__len__() - self.memory + index
        for i in range(self.memory + 1):
            inputs[i] = super().__getitem__((index + i) % super().__len__())[0].real
            inputs[i + self.memory + 1] = (
                super().__getitem__((index + i) % super().__len__())[0].imag
            )
        target[0] = (
            super().__getitem__((index + self.memory) % super().__len__())[1].real
        )
        target[1] = (
            super().__getitem__((index + self.memory) % super().__len__())[1].imag
        )
        return inputs, target


class RVTDNN(nn.Module):
    r"""
    The real-valued time-delay neural network implementation in PyTorch.

    Reference:
    [Dynamic Behavioral Modeling of 3G Power Amplifiers Using Real-Valued Time-Delay Neural Networks](http://ieeexplore.ieee.org/document/1273746/)

    Parameters:
    - layer_dims: The network structure, e.g. [6, 32, 32, 2]
    - activation: The activation function, e.g. "ReLU", "Tanh", "ELU" or "None"
    """

    def __init__(self, layer_dims, activation="ReLU"):
        super().__init__()
        assert activation == "ReLU" or "Tanh" or "ELU" or "None"
        self.layers = nn.Sequential()
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim))
            if activation == "ReLU":
                self.layers.add_module("actFunc " + str(index), nn.ReLU())
            elif activation == "Tanh":
                self.layers.add_module("actFunc " + str(index), nn.Tanh())
            elif activation == "ELU":
                self.layers.add_module("actFunc " + str(index), nn.ELU(alpha=1))
            elif activation == "None":
                pass
        if activation != "None":
            self.layers = self.layers[:-1]  # remove the last activation layer

    def forward(self, x):
        return self.layers(x)
