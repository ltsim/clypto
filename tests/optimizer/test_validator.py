#!/usr/bin/env python
# Created by "Thieu" at 07:57, 16/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import pytest

import clypto as cy


@pytest.mark.parametrize(
    "value, bound, output",
    [
        (-3.3, [-10, 10], -3),
        (1000, (2, float("inf")), 1000),
        (0.5, [0.3, 2], 0),
    ],
)
def test_check_bound(value, bound, output):
    assert cy.validator(int, value, bound, "value") == output


@pytest.mark.parametrize(
    "value, bound, output",
    [
        (None, [-10, 10], 0),
        ("hello", (2, float("inf")), 0),
        (-0.22, [0.3, 2], 0),
        ([3, 2], (2, float("inf")), 0),
        ((4, 2), [0.3, 2], 0),
    ],
)
def test_check_float(value, bound, output):
    with pytest.raises(TypeError) as e:
        cy.validator(float, value, bound, "value")
    assert e.type == TypeError
