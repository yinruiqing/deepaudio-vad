import numpy as np
import torch
from torch.utils.data import DataLoader


class VadUttDataLoader(DataLoader):
    r"""
    Text Data Loader

    Args:
        dataset (torch.utils.data.Dataset): dataset from which to load the data.
        num_workers (int): how many subprocesses to use for data loading.
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            num_workers: int,
            batch_size: int,
            **kwargs,
    ) -> None:
        super(VadUttDataLoader, self).__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            **kwargs,
        )