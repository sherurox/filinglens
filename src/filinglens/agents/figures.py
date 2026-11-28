"""Does a figure, as an answer writes it, match a reference value? (FL-M6 T6)

Hand-written by me (FL-M6 live-coding piece). The numbers checker (agents/numbers.py) finds each
figure in an answer ("$1.23 billion", "12.5%", "-$0.4 billion"), turns it into the four arguments
below, and calls figure_matches() against the reference values from marts.fct_financials (both
the last-reported and the first-reported value). An answer with a figure that matches nothing is
blocked.
"""

import math
from fractions import Fraction


def _as_written(x: float) -> Fraction:
    """The shortest decimal that round-trips to x, as an exact fraction (0.1 -> 1/10)."""
    return Fraction(repr(float(x)))


def figure_matches(stated: float, decimals: int, scale: float, actual: float) -> bool:
    """True if `actual` rounds to the figure as it is written.

    stated:   the number as written, with its sign: 1.23 for "$1.23 billion", -0.4 for
              "-$0.4 billion"
    decimals: digits written after the decimal point: 2 for "1.23", 0 for "812"
    scale:    what the unit multiplies by: 1e9 "billion", 1e6 "million", 1.0 plain dollars,
              0.01 for a percent ("12.5%" -> stated 12.5, decimals 1, scale 0.01)
    actual:   the reference value in base units (USD, or a fraction such as 0.125 for a margin)

    Rule: a match means `actual` lies within half a unit of the last written digit, ends
    included. "$1.2 billion" matches any value from 1.15e9 to 1.25e9.

    Raises ValueError if decimals is not an int (bool included) or is < 0, if scale <= 0, or if
    stated, scale or actual is not finite (NaN, infinity).
    """
    # bool is a subclass of int, so True would pass as 1 without the first check
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise ValueError(f"decimals must be an int >= 0, got {decimals!r}")
    if not scale > 0:  # `not >` so NaN is rejected too
        raise ValueError(f"scale must be > 0, got {scale!r}")
    for name, value in (("stated", stated), ("scale", scale), ("actual", actual)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    # Exact Fraction math on each float's shortest decimal, so on-paper ends
    # (0.155 for "15%") stay in.
    center = _as_written(stated) * _as_written(scale)
    half_unit = _as_written(scale) / (2 * 10**decimals)
    return abs(_as_written(actual) - center) <= half_unit
