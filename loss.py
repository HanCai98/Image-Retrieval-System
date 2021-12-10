import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class MatrixSoftmaxCELoss(nn.Module):
    def __init__(self, device):
        super(MatrixSoftmaxCELoss, self).__init__()
        # self.softmax_temperature = nn.Parameter(torch.tensor(0.07))
        self.softmax_temperature = 0.07
        self.device = device

    def forward(self, matrix):
        batchsize = matrix.shape[0]
        labels = torch.arange(batchsize).to(self.device)
        row_matrix = F.log_softmax(matrix / self.softmax_temperature, dim=1)
        col_matrix = F.log_softmax(matrix / self.softmax_temperature, dim=0).T
        loss = F.nll_loss(row_matrix, labels) + F.nll_loss(col_matrix, labels)
        loss = F.nll_loss(row_matrix, labels)
        return torch.sum(loss)


class ContrastiveLoss(nn.Module):
    def __init__(self, device, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, matrix):
        batchsize = matrix.shape[0]
        positive_indices = torch.arange(batchsize).to(self.device)
        negative_candidate_indices = torch.argsort(matrix, descending=True)[:, :2]
        negative_is_first = torch.ne(negative_candidate_indices[:, 0], positive_indices)
        negative_indices = torch.where(negative_is_first, negative_candidate_indices[:, 0], negative_candidate_indices[:, 1])
        
        positives = matrix[positive_indices, positive_indices]
        negatives = matrix[positive_indices, negative_indices]
        loss = torch.clamp(negatives - positives + self.margin, min=0)

        return torch.sum(loss)

