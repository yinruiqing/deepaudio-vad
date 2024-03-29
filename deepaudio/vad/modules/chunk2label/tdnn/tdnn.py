from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class StatsPool(nn.Module):
    """Calculate pooling as the concatenated mean and standard deviation of a sequence"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, hidden_size)
            A batch of sequences.
        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, 2 * hidden_size)
        """
        mean, std = torch.mean(x, dim=1), torch.std(x, dim=1)
        return torch.cat((mean, std), dim=1)


class TDNN(nn.Module):
    def __init__(
            self,
            context: list,
            input_channels: int,
            output_channels: int,
            full_context: bool = True,
    ):
        """
        Implementation of a 'Fast' TDNN layer by exploiting the dilation argument of the PyTorch Conv1d class
        Due to its fastness the context has gained two constraints:
            * The context must be symmetric
            * The context must have equal spacing between each consecutive element
        For example: the non-full and symmetric context {-3, -2, 0, +2, +3} is not valid since it doesn't have
        equal spacing; The non-full context {-6, -3, 0, 3, 6} is both symmetric and has an equal spacing, this is
        considered valid.
        :param context: The temporal context
        :param input_channels: The number of input channels
        :param output_channels: The number of channels produced by the temporal convolution
        :param full_context: Indicates whether a full context needs to be used
        """
        super(TDNN, self).__init__()
        self.full_context = full_context
        self.input_dim = input_channels
        self.output_dim = output_channels

        context = sorted(context)
        self.check_valid_context(context, full_context)

        if full_context:
            kernel_size = context[-1] - context[0] + 1 if len(context) > 1 else 1
            self.temporal_conv = weight_norm(
                nn.Conv1d(input_channels, output_channels, kernel_size)
            )
        else:
            # use dilation
            delta = context[1] - context[0]
            self.temporal_conv = weight_norm(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=len(context),
                    dilation=delta,
                )
            )

    def forward(self, x):
        """
        :param x: is one batch of data, x.size(): [batch_size, sequence_length, input_channels]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, len(valid_steps), output_dim]
        """
        x = self.temporal_conv(torch.transpose(x, 1, 2))
        return F.relu(torch.transpose(x, 1, 2))

    @staticmethod
    def check_valid_context(context: list, full_context: bool) -> None:
        """
        Check whether the context is symmetrical and whether and whether the passed
        context can be used for creating a convolution kernel with dil
        :param full_context: indicates whether the full context (dilation=1) will be used
        :param context: The context of the model, must be symmetric if no full context and have an equal spacing.
        """
        if full_context:
            assert (
                    len(context) <= 2
            ), "If the full context is given one must only define the smallest and largest"
            if len(context) == 2:
                assert context[0] + context[-1] == 0, "The context must be symmetric"
        else:
            assert len(context) % 2 != 0, "The context size must be odd"
            assert (
                    context[len(context) // 2] == 0
            ), "The context contain 0 in the center"
            if len(context) > 1:
                delta = [context[i] - context[i - 1] for i in range(1, len(context))]
                assert all(
                    delta[0] == delta[i] for i in range(1, len(delta))
                ), "Intra context spacing must be equal!"


class TDNNNet(nn.Module):
    def __init__(self, configs):
        super(TDNNNet, self).__init__()
        frame1 = TDNN(
            context=[-2, 2],
            input_channels=configs.feature.n_mels,
            output_channels=32,
            full_context=True,
        )
        frame2 = TDNN(
            context=[-2, 0, 2],
            input_channels=32,
            output_channels=32,
            full_context=False,
        )
        frame3 = TDNN(
            context=[-3, 0, 3],
            input_channels=32,
            output_channels=32,
            full_context=False,
        )
        self.tdnn = nn.Sequential(frame1, frame2, frame3, StatsPool())
        self.final_layer_ = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.tdnn(x)
        return F.sigmoid(x)
