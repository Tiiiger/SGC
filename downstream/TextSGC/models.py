import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class SGC(nn.Module):
    def __init__(self, nfeat, nclass, bias=False):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass, bias=bias)
        torch.nn.init.xavier_normal_(self.W.weight)

    def forward(self, x):
        out = self.W(x)
        return out
