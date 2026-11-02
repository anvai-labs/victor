"""Literal precision must survive parsing, midpoint rounding and extreme exponents."""

import math
import random
import struct

import pytest

from victor.core.graph_rag.numeric_literals import typed_float_value


def test_decimal_just_above_midpoint_avoids_double_rounding():
    raw = "1.0000000596046447753906250000000000001"
    assert struct.unpack("!f", struct.pack("!f", float(raw)))[0] == 1.0
    assert typed_float_value(raw, 32) == float.fromhex("0x1.000002p0")
    assert typed_float_value("1.000000059604644775390625", 32) == 1.0


def test_midpoints_underflow_and_signed_zero():
    assert typed_float_value("16777217", 32) == 16777216
    assert typed_float_value("16777219", 32) == 16777220
    assert typed_float_value("1e-45", 32) == math.ldexp(1, -149)
    assert typed_float_value("1e-9999999", 32) == 0
    assert math.copysign(1, typed_float_value("-0.0", 32)) == -1


def test_binary32_roundtrips_agree_with_independent_struct_reference():
    rng = random.Random(20261102)
    for _ in range(1000):
        bits = rng.randrange(0x7F800000)
        value = struct.unpack("!f", struct.pack("!I", bits))[0]
        assert typed_float_value(repr(value), 32) == value


@pytest.mark.parametrize(
    "text,width,reason",
    [
        ("1e100", 32, "nonfinite"),
        ("1e99999", 64, "nonfinite"),
        ("NaN", 32, "nonfinite"),
        ("1" * 4097, 32, "budget"),
        ("garbage", 32, "numeric"),
        ("1.0", 16, "width"),
    ],
)
def test_unrepresentable_or_unbounded_literals_are_rejected(text, width, reason):
    with pytest.raises(ValueError, match=reason):
        typed_float_value(text, width)


def test_ambient_decimal_precision_does_not_change_fingerprints():
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 4
        assert typed_float_value("1.0000000596046447753906250000000000001", 32) == float.fromhex(
            "0x1.000002p0"
        )
        assert typed_float_value("0.100000001490116119384765625", 32) == 0.10000000149011612


@pytest.mark.parametrize("text,sign", [("0e39", 1), ("-0e39", -1), ("0.0e1000", 1)])
def test_large_zero_exponent_is_finite(text, sign):
    value = typed_float_value(text, 32)
    assert value == 0
    assert math.copysign(1, value) == sign
