"""Exact decimal-to-binary32 literal rounding without a binary64 intermediate."""

from decimal import Decimal, InvalidOperation
import math


def typed_float_value(text: str, width: int) -> float:
    """Return the finite literal value, preserving ties-to-even binary32 rounding.

    Decimal source may lie just to one side of a binary32 midpoint. Converting it
    through Python's binary64 float first can erase that distinction. Integer
    rounding below preserves it, including subnormals; work is bounded by spelling.
    """
    if width not in {32, 64}:
        raise ValueError("numeric-literal-width")
    if len(text) > 4096:
        raise ValueError("numeric-literal-budget")
    if width == 64:
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError("numeric-literal") from exc
        if not math.isfinite(value):
            raise ValueError("nonfinite-literal")
        return value
    try:
        decimal = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("numeric-literal") from exc
    if not decimal.is_finite():
        raise ValueError("nonfinite-literal")
    sign = -1 if decimal.is_signed() else 1
    if decimal.is_zero():
        return math.copysign(0.0, sign)
    if decimal.adjusted() > 38:
        raise ValueError("nonfinite-literal")
    if decimal.adjusted() < -46:
        return math.copysign(0.0, sign)
    numerator, denominator = decimal.copy_abs().as_integer_ratio()
    exponent = numerator.bit_length() - denominator.bit_length()
    if (
        numerator < denominator << exponent
        if exponent >= 0
        else numerator << -exponent < denominator
    ):
        exponent -= 1
    quantum = max(exponent - 23, -149)
    numerator <<= max(-quantum, 0)
    denominator <<= max(quantum, 0)
    rounded, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and rounded % 2):
        rounded += 1
    value = math.ldexp(float(rounded), quantum)
    if value > math.ldexp(float((1 << 24) - 1), 104):
        raise ValueError("nonfinite-literal")
    return math.copysign(value, sign)
