import torch
from torch.utils.data import Dataset
from scipy.io import loadmat


class PADataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        pa_input=0,
        pa_output=0,
        train_ratio=0.8,
        train=True,
        inverse=False,
    ):
        """
        Arguments:
            root_dir (string): Directory with MATLAB file.
            train: Divided into train set or validation set.
            inverse: Inverse modeling (DPD) or not.
        """
        self.train = train
        self.train_ratio = train_ratio
        if root_dir is not None:
            data = loadmat(root_dir)
            pa_input = torch.tensor(data["xorg"].reshape(-1))
            pa_output = torch.tensor(data["yorg"].reshape(-1))
        if train:
            self.pa_input = pa_input
            self.pa_output = pa_output
        else:
            self.pa_input = torch.roll(pa_input, -int(len(pa_input) * train_ratio))
            self.pa_output = torch.roll(pa_output, -int(len(pa_input) * train_ratio))
        if inverse:
            self.pa_input, self.pa_output = (
                self.pa_output.clone(),
                self.pa_input.clone(),
            )

    def __len__(self):
        return len(self.pa_input)

    def __getitem__(self, index):
        # turn numpy to tensor
        return self.pa_input[index], self.pa_output[index]

    def getseries(self):
        return self.pa_input, self.pa_output
