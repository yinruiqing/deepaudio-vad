import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from .. import register_criterion
from ..cross_entropy.configuration import BinaryCrossEntropyLossConfigs


@register_criterion("cross_entropy", dataclass=CrossEntropyLossConfigs)
class CrossEntropyLoss(nn.Module):
    def __init__(
            self,
            configs: DictConfig,
    ) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.binary_cross_entropy_loss = nn.BCELoss(
            reduction=configs.criterion.reduction,
        )

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = predictions.flatten()
        targets = targets.flatten()

        return self.cross_entropy_loss(
            predictions,
            targets,
        )
