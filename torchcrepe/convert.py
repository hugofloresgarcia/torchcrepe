import scipy
import torch
from typing import List

import torchcrepe


###############################################################################
# Pitch unit conversions
###############################################################################


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = torchcrepe.CENTS_PER_BIN * bins + 1997.3794084376191

    # Trade quantization error for noise
    return dither(cents)


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents: torch.Tensor, quantize_fn: str = 'floor') -> int:
    """Converts cents to pitch bins"""
    bins = (cents - 1997.3794084376191) / torchcrepe.CENTS_PER_BIN
    assert quantize_fn in ('floor', 'ceil')
    result = torch.floor(bins) if quantize_fn=="floor" else torch.ceil(bins)
    return result.int()


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn: str='floor'):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * torch.log2(frequency / 10.)


###############################################################################
# Utilities
###############################################################################


def dither(cents:torch.Tensor):
    """Dither the predicted pitch in cents to remove quantization error"""
    loc: int=20
    scale: int=2 * loc
    size: List[int]=cents.size()

    noise: torch.Tensor = torch.rand(size)
    noise = (noise - 0.5) * 2
    noise = noise * scale + loc
                                   
    return cents + noise
