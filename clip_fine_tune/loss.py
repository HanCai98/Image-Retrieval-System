import torch
import numpy as np

class InfoNCELoss(torch.nn.Module):
    def __init__(self, device):
        super(InfoNCELoss, self).__init__()
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, inputs_rows, inputs_cols):
        batchsize = inputs_rows.shape[0]
        labels = torch.arange(batchsize).to(self.device)
        # loss = (self.CELoss(inputs_rows, labels) + self.CELoss(inputs_cols, labels)) / 2
        loss = self.CELoss(inputs_cols, labels)
        return torch.mean(loss)

