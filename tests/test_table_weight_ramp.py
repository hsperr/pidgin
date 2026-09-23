"""E49's table-result ramp: off by default, linear and clamped when it is on."""
import argparse

import pytest

from bridgezero.fourseat.train import table_weight_at


def args(**kw) -> argparse.Namespace:
    base = {"table_weight": 1.0, "table_weight_start": 0.0, "table_weight_steps": 0}
    return argparse.Namespace(**{**base, **kw})


def test_constant_without_a_ramp():
    """Every run before E49 passed no ramp flags and must be unaffected."""
    a = args()
    assert table_weight_at(1, a) == 1.0
    assert table_weight_at(20000, a) == 1.0
    assert table_weight_at(1, args(table_weight=0.25)) == 0.25


def test_ramp_is_linear_between_the_ends():
    a = args(table_weight_steps=1000)
    assert table_weight_at(1, a) == pytest.approx(0.0)
    assert table_weight_at(251, a) == pytest.approx(0.25)
    assert table_weight_at(501, a) == pytest.approx(0.5)
    assert table_weight_at(1001, a) == pytest.approx(1.0)


def test_ramp_holds_at_the_target_afterwards():
    a = args(table_weight_steps=1000)
    assert table_weight_at(5000, a) == pytest.approx(1.0)
    assert table_weight_at(20000, a) == pytest.approx(1.0)


def test_ramp_can_start_above_zero_and_fall():
    a = args(table_weight=0.0, table_weight_start=1.0, table_weight_steps=100)
    assert table_weight_at(1, a) == pytest.approx(1.0)
    assert table_weight_at(51, a) == pytest.approx(0.5)
    assert table_weight_at(101, a) == pytest.approx(0.0)


def test_resumed_step_before_the_start_is_clamped():
    """--resume can re-enter the loop at any step; the fraction must stay in [0, 1]."""
    a = args(table_weight_steps=1000)
    assert table_weight_at(0, a) == pytest.approx(0.0)
    assert table_weight_at(-5, a) == pytest.approx(0.0)
