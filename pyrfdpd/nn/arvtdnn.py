import torch
from . import rvtdnn


class Dataset(rvtdnn.Dataset):
    def __init__(
        self,
        root_dir=None,
        pa_input=0,
        pa_output=0,
        train_ratio=0.8,
        memory=3,
        order=3,
        order_memory=2,
        train=True,
        inverse=False,
    ):
        """order_memory is the memory of augumented terms"""
        assert order >= 1
        super().__init__(
            root_dir, pa_input, pa_output, train_ratio, memory, train, inverse
        )
        self.order = order
        self.order_memory = order_memory

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        # Data array pattern
        # Except the items in RVTDNN, add augmented terms
        # for example : order = 3, order_memory = 1
        # |x(0)|
        # |x(0)|^2
        # |x(0)|^3
        # |x(1)|
        # |x(1)|^2
        # |x(1)|^3
        inputs, target = super().__getitem__(index)
        for j in range(self.order_memory + 1):
            amplitude_org = torch.abs(
                torch.complex(inputs[j], inputs[j + self.memory + 1])
            )
            amplitude = torch.zeros(self.order)
            amplitude[0] = amplitude_org
            for i in range(2, self.order + 1):
                amplitude[i - 1] = torch.pow(amplitude_org, i)
            inputs = torch.cat((inputs, amplitude))
        return inputs, target


ARVTDNN = rvtdnn.RVTDNN
