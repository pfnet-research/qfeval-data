import gc as _gc
import hashlib
import logging
import math
import typing

import numpy as np
import torch

logger = logging.getLogger(__name__)


# T represents a generic type.
T = typing.TypeVar("T")
# Array represents a generic NumPy-array-like type.  Types specified here must
# be supported by ArrayUtil.
Array = typing.TypeVar("Array", torch.Tensor, np.ndarray)


################################################################################
# Array functions
################################################################################


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Returns a numpy object corresponding to the given Tensor.  This should
    work if the Tensor object has a gradient or it is placed on a GPU."""
    return typing.cast(np.ndarray, tensor.detach().cpu().numpy())


def nans(
    shape: typing.Optional[typing.Tuple[int, ...]] = None,
    like: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns a tensor of NaNs."""
    if like is None:
        raise ValueError("like must be specified.")
    if shape is None:
        shape = like.shape
    result = torch.as_tensor(math.nan, dtype=like.dtype, device=like.device)
    return result.expand(shape)


def make_array_mapping(
    ref: np.ndarray, like: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Returns two arrays representing a mapping from `like` to `ref`.  The
    first array represents indexes, each of which indicates that a value in
    `like` should be an index of `ref` (i.e., `ref[indexes[i]] == like[i]`).
    The second array represents a mask, each of which represents whether or
    not the corresponding index value should be masked (i.e. invalid).
    """
    indexes = np.searchsorted(ref, like)
    mask = np.equal(indexes, np.searchsorted(ref, like, side="right"))
    # Prevent out-of-bounds error.  A large value in ref would cause 1 + an
    # index number of the last element, and it causes IndexError in some
    # usage.
    indexes[mask] = 0
    return indexes, mask


def are_broadcastable_shapes(
    x: typing.Union[typing.Tuple[int, ...], torch.Size],
    *ys: typing.Union[typing.Tuple[int, ...], torch.Size],
) -> bool:
    r"""Returns true iff the given shapes are broadcastable for numpy-like
    operations (including PyTorch, CuPy).
    """
    try:
        # If broadcasting fails, it raises a ValueError.
        _ = np.broadcast_shapes(x, *ys)
    except ValueError:
        return False
    return True


################################################################################
# Time-related functions
################################################################################


@typing.overload
def ceil_time(
    t: np.datetime64,
    d: np.timedelta64,
    origin: typing.Optional[np.datetime64] = None,
    offset: typing.Optional[np.timedelta64] = None,
) -> np.datetime64:
    pass


@typing.overload
def ceil_time(
    t: np.ndarray,
    d: np.timedelta64,
    origin: typing.Optional[np.datetime64] = None,
    offset: typing.Optional[np.timedelta64] = None,
) -> np.ndarray:
    pass


def ceil_time(
    t: typing.Union[np.datetime64, np.ndarray],
    d: np.timedelta64,
    origin: typing.Optional[np.datetime64] = None,
    offset: typing.Optional[np.timedelta64] = None,
) -> typing.Union[np.datetime64, np.ndarray]:
    if offset is not None:
        ret: typing.Union[np.datetime64, np.ndarray] = ceil_time(
            t.astype("datetime64[us]") - offset.astype("timedelta64[us]"),
            d,
            origin=origin,
        )
        return ret
    floor_t = floor_time(t, d, origin=origin)
    divisible = np.equal(
        t.astype("datetime64[us]"), floor_t.astype("datetime64[us]")
    )
    return typing.cast(
        typing.Union[np.datetime64, np.ndarray],
        floor_t + d * (~divisible).astype(int),
    )


@typing.overload
def floor_time(
    t: np.datetime64,
    d: np.timedelta64,
    origin: typing.Optional[np.datetime64] = None,
    offset: typing.Optional[np.timedelta64] = None,
) -> np.datetime64:
    pass


@typing.overload
def floor_time(
    t: np.ndarray,
    d: np.timedelta64,
    origin: typing.Optional[np.datetime64] = None,
    offset: typing.Optional[np.timedelta64] = None,
) -> np.ndarray:
    pass


def floor_time(
    t: typing.Union[np.datetime64, np.ndarray],
    d: np.timedelta64,
    origin: typing.Optional[np.datetime64] = None,
    offset: typing.Optional[np.timedelta64] = None,
) -> typing.Union[np.datetime64, np.ndarray]:
    if offset is not None:
        ret: typing.Union[np.datetime64, np.ndarray] = floor_time(
            t.astype("datetime64[us]") - offset.astype("timedelta64[us]"),
            d,
            origin,
        )
        return ret
    if origin is None:
        return floor_time(t, d, time_origin(d))
    unit, di = np.datetime_data(d)
    di = di * d.astype(int)
    origin = origin.astype(f"datetime64[{unit}]")
    ti: np.ndarray = (t.astype(f"datetime64[{unit}]") - origin).astype(int)
    return typing.cast(
        typing.Union[np.datetime64, np.ndarray],
        (ti // di * di).astype(f"timedelta64[{unit}]") + origin,
    )


def time_origin(d: np.timedelta64) -> np.datetime64:
    unit, _ = np.datetime_data(d)
    if unit in ("M", "Y"):
        # qfeval uses 1000-01-01 as the datetime origin for monthly/yearly
        # ticks so as to show multiples of 10/100/1000 years for a long term
        # x-axis.
        ret: np.datetime64 = np.datetime64("1000-01-01", unit)
        return ret
    # qfeval uses 1893-01-01 as the datetime origin because of the
    # following reasons: (1) it is a first day of a year (2) it is on
    # Sunday, (3) it is older than 1896, in which Done Jones Indutrial
    # Index starts, (3) 2^63 nanoseconds from 1893-01-01 represent enough
    # future (2185-04-11).  This intentionally uses a week starting from
    # Sunday while the ISO week date system uses weeks starting from
    # Monday.  Since the real market starts trading before 00:00 on Monday
    # in UTC (e.g., a forex market actually starts at 22:00 on Sunday in
    # UTC), and splitting weeks using 00:00 on Sunday in UTC is more
    # realistic.
    ret = np.datetime64("1893-01-01", unit)
    return ret


################################################################################
# Other functions
################################################################################


def sha1(x: typing.Union[bytes, str, np.ndarray, torch.Tensor]) -> str:
    if isinstance(x, torch.Tensor):
        return sha1(to_numpy(x))
    if isinstance(x, np.ndarray):
        # NOTE: This blends the hash of shape information because
        # ndarray.tobytes discards shape information.
        return sha1(sha1(np.array(x.shape).tobytes()) + sha1(x.tobytes()))
    if isinstance(x, str):
        return sha1(x.encode("utf-8"))
    m = hashlib.sha1()
    m.update(x)
    return m.hexdigest()


def gc() -> None:
    r"""Runs garbage collection.  This also releases GPU memory if any."""
    _gc.collect(2)
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()


def torch_device(device: typing.Any) -> torch.device:
    r"""Returns a corresponding torch.device.

    This supports "auto" in addition to the PyTorch device resolver.  "auto"
    chooses the first GPU if available.  Otherwise, it falls back to CPU.
    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch_device("cpu")
    if device == "auto":
        if torch.cuda.device_count() > 0:
            return torch_device("cuda")
        logger.warning("Falling back to the CPU device.")
        return torch_device("cpu")
    return torch.device(device)
