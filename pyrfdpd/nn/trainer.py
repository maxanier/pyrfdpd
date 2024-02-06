import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pdb


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        name: str = "RVTDNN",
        lr: float = 0.001,
        batch_size: int = 128,
        lossFcn=nn.MSELoss(),
        optimizer=optim.Adam,
        tensorboard: bool = False,
        logger=None,
    ) -> None:
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.net = net.to(self.device)
        self.lossFcn = lossFcn
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer(self.net.parameters(), self.lr)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if logger:
            self.logger = logger
            self.logger.debug(f"Training on device {self.device}.")
        else:
            print(f"Training on device {self.device}.")

        if tensorboard:
            self.writer = SummaryWriter("runs/" + name + "_{}".format(self.timestamp))
            self.writer.add_text(
                "Num of Parameters",
                str(sum([p.data.nelement() for p in net.parameters()])),
                0,
            )
            rand_input = torch.randn(next(net.parameters()).size(1))
            rand_input = torch.stack((rand_input, rand_input))
            self.writer.add_graph(self.net, rand_input.to(self.device))
            self.writer.flush()

    def train(self, training_set, validation_set, epochs=1000, patiences=10):
        training_loader = DataLoader(training_set, batch_size=self.batch_size)
        validation_loader = DataLoader(validation_set, batch_size=self.batch_size)
        patience = 0
        last_vloss = 10
        for epoch in range(epochs):
            self.net.train()
            for data in training_loader:
                inputs, target = data
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.lossFcn(outputs, target)
                loss.backward()
                self.optimizer.step()

            running_vloss = 0.0
            self.net.eval()
            with torch.no_grad():
                for idx, vdata in enumerate(validation_loader):
                    vinputs, vtarget = vdata
                    vinputs = vinputs.to(self.device)
                    vtarget = vtarget.to(self.device)
                    voutputs = self.net(vinputs)
                    vloss = self.lossFcn(voutputs, vtarget)
                    running_vloss += vloss.item()

            avg_loss = loss.item()  # loss of the last batch
            avg_vloss = running_vloss / (idx + 1)  # average loss of the validation set

            if hasattr(self, "writer"):
                self.writer.add_scalars(
                    "Training vs. Validation Loss",
                    {"Training": avg_loss, "Validation": avg_vloss},
                    epoch + 1,
                )

            # Eearly stopping
            if avg_vloss > last_vloss:
                patience += 1
                if patience > patiences:
                    break
            else:
                patience = 0
            last_vloss = avg_vloss

            if self.logger:
                self.logger.debug(
                    f"epoch{epoch+1:4d}, loss: train {avg_loss:.6f}, validation {avg_vloss:.6f}, patience {patience}"
                )
            else:
                print(
                    f"epoch{epoch+1:4d}, loss: train {avg_loss:.6f}, validation {avg_vloss:.6f}, patience {patience}"
                )
        if self.logger:
            self.logger.debug("Finished Training!")
        else:
            print("Finished Training!")

    def predict(self, data_set):
        data_loader = DataLoader(data_set, batch_size=self.batch_size)
        # predict_data is in GPU
        predict_data = torch.empty(0).to(self.device)
        # target_data is in CPU
        target_data = torch.empty(0)
        with torch.no_grad():
            for data in data_loader:
                inputs, target = data
                outputs = self.net(inputs.to(self.device))
                predict_data = torch.cat(
                    (
                        predict_data,
                        torch.complex(outputs[:, 0], outputs[:, 1]).flatten(),
                    ),
                    dim=0,
                )
                target_data = torch.cat(
                    (target_data, torch.complex(target[:, 0], target[:, 1]).flatten()),
                    dim=0,
                )
        return predict_data.cpu()
