import torch
from . import rvtdnn


Dataset = rvtdnn.Dataset


class R2TDNN(rvtdnn.RVTDNN):
    """ """

    def __init__(self, layer_dims, activation):
        super().__init__(layer_dims=layer_dims, activation=activation)
        self.memory = layer_dims[0] // 2

    def forward(self, x):
        # The x is ranged like [i0, i1, i2, ..., q0, q1, q2, ...]
        out = self.layers(x) + self.shortcut(x)
        return out

    def shortcut(self, x):
        return torch.stack([x[:, self.memory - 1], x[:, -1]], dim=1)

