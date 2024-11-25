from typing import Tuple

import torch
from torch import nn

Tensor = torch.Tensor


class TripletLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super(TripletLoss, self).__init__()
        self.temperature = temperature

    def logsumexp(self, dots: tuple[Tensor]) -> Tensor:
        return torch.log(torch.sum(torch.exp(torch.stack(dots) / self.temperature), dim=0))

    def log_softmax(self, dots: tuple[Tensor]) -> Tensor:
        return dots[0] / self.temperature - self.logsumexp(dots)

    def cross_entropy_loss(self, dots: tuple[Tensor]) -> Tensor:
        return torch.mean(-self.log_softmax(dots))

    def forward(self, dots: tuple[Tensor]):
        return self.cross_entropy_loss(dots)
