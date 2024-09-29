# модель и функция потерь работают с таргетом типа float, а для метрики нужен int
import torch
import torchmetrics
from torch import nn


class AUROC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.metric = torchmetrics.AUROC(task='binary', num_labels=1)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.metric(preds, target.int())  # we are here for this line

    def compute(self) -> torch.Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()


class F1_weighted(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.metric = torchmetrics.classification.F1Score("multiclass", num_classes=num_labels, average="weighted")

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.metric(preds.argmax(dim=-1), target.argmax(dim=-1))  # we are here for this line

    def compute(self) -> torch.Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()