from dataclasses import dataclass, field

from ...dataclass.configurations import DeepMMDataclass


@dataclass
class BinaryCrossEntropyLossConfigs(DeepMMDataclass):
    criterion_name: str = field(
        default="binary_cross_entropy", metadata={"help": "Criterion name for training"}
    )
    reduction: str = field(
        default="mean", metadata={"help": "Reduction method of criterion"}
    )

