import numpy as np
import torch

from . import util
from .data import Data


class Flattener(object):
    r"""Flattener assists conversion between Data and torch.Tensor.
    This assumes that Data should manage a table with timestamp/symbol indexes
    and torch.Tensor should manage an array with a batch index representing a
    subset of timestamp/symbol pairs.

    Args:
        - data (Data): A reference table representing which timestamp/symbol
          pairs should be flattened.
    """

    def __init__(self, *data: Data):
        assert len(data) > 0, "One or more Data must be given."
        # Memorize timestamp/symbol indexes.
        self.__timestamps = data[0].timestamps
        self.__symbols = data[0].symbols
        # Create a mask representing which pairs of timestamp/symbol should
        # be flattened.
        result = torch.as_tensor(False, device=data[0].device)
        for d in data:
            assert np.array_equal(d.timestamps, self.__timestamps)
            assert np.array_equal(d.symbols, self.__symbols)
            for v in d.tensors.values():
                v = torch.isnan(v).reshape(v.shape[:2] + (-1,))
                result = v.any(dim=2) | result
        self.__mask = ~result
        self.__count = int(self.__mask.sum())

    def flatten(self, data: Data) -> torch.Tensor:
        r"""Flattens timestamp/symbol indexes as a single batch index.

        Parameters:
            - data (Data): a Data object having a tensor to be flattened.

        Shape:
            - data: :math:`(B, *)` where `*` means any number of additional
              dimensions.
            - return: :math:`(T, S, *)` where `*` means the same additional
              dimensions as data has.
        """
        assert np.array_equal(data.timestamps, self.__timestamps)
        assert np.array_equal(data.symbols, self.__symbols)
        t = data.tensor
        t = t.reshape((-1,) + t.shape[2:])
        return t[self.__mask.reshape(-1)]

    def timestamp_indexes(self) -> torch.Tensor:
        timestamp_indexes = torch.arange(
            self.__mask.shape[0],
            device=self.__mask.device,
        )[:, None].expand(self.__mask.shape)
        return timestamp_indexes.flatten()[self.__mask.flatten()]

    def symbol_indexes(self) -> torch.Tensor:
        symbol_indexes = torch.arange(
            self.__mask.shape[1],
            device=self.__mask.device,
        )[None, :].expand(self.__mask.shape)
        return symbol_indexes.flatten()[self.__mask.flatten()]

    def unflatten(self, tensor: torch.Tensor, name: str = "") -> Data:
        r"""Unflattens a batch index to timestamp/symbol indexes.

        This restores the original dimension based on the Data object given in
        initialization.  The returned Data should have a column name specified
        by `name`.

        Parameters:
            - tensor (torch.Tensor): a tensor to be unflattened.
            - name (str): the column name of the Data to be returned.
        """
        assert (
            tensor.shape[0] == self.__count
        ), f"""Invalid batch size: actual={
            tensor.shape[0]}, exepcted={self.__count}"""
        # Extract an extra shape of the given tensor.
        extra_shape = tensor.shape[1:]
        # Prepare the result tensor with NaN values.
        result = util.nans(self.__mask.shape + extra_shape, like=tensor)
        # Create a mask for scatter.
        mask = self.__mask.reshape(self.__mask.shape + (1,) * len(extra_shape))
        mask = mask.expand(self.__mask.shape + extra_shape).to(tensor.device)
        # Scatter the given tensor to the result object.
        result = result.masked_scatter(mask, tensor.reshape(-1))
        return Data.from_tensors(
            {name: result}, self.__timestamps, self.__symbols
        )
