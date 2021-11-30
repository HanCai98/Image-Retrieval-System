import torch
import numpy as np

class InfoNCELoss(torch.nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()
        self.CELoss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs_rows, inputs_cols):
        batchsize = inputs_rows.shape[0]
        labels = torch.arange(batchsize)
        loss = (self.CELoss(inputs_rows, labels) + self.CELoss(inputs_cols, labels)) / 2
        return torch.mean(loss)
