import torch
import numpy as np
from torch import nn


class MatrixSoftmaxCELoss(nn.Module):
    def __init__(self):
        super(MatrixSoftmaxCELoss, self).__init__()
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.softmax_temperature = nn.Parameter(torch.tensor(1))

    def forward(self, matrix):
        batchsize = matrix.shape[0]
        labels = torch.arange(batchsize).to(self.device)
        row_matrix = torch.softmax(matrix / self.softmax_temperature, dim=1)
        col_matrix = torch.softmax(matrix / self.softmax_temperature, dim=0).T
        loss = (self.CELoss(row_matrix, labels) + self.CELoss(col_matrix, labels)) / 2
        return torch.mean(loss)
