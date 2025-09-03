import math

import pytest

from audio_toolset.util import (
    convert_db_to_factor,
    convert_linear_to_db,
    convert_power_to_db,
)

TOLERANCE = 1e-3


@pytest.mark.parametrize(
    "db, factor",
    [
        (-3, 0.7079),
        (0, 1.0),
        (6, 1.9952),
    ],
)
def test_convert_db_to_factor(db: float, factor: float) -> None:
    assert math.isclose(
        convert_db_to_factor(db), factor, rel_tol=TOLERANCE, abs_tol=TOLERANCE
    )


@pytest.mark.parametrize(
    "db, power",
    [
        (-3.0103, 0.5),
        (0, 1),
        (10, 10),
    ],
)
def test_convert_power_to_db(db: float, power: float) -> None:
    assert math.isclose(
        convert_power_to_db(power), db, rel_tol=TOLERANCE, abs_tol=TOLERANCE
    )


@pytest.mark.parametrize(
    "linear, db",
    [
        (0.7079, -3),
        (1.0, 0),
        (1.9952, 6),
        (0, -240),
    ],
)
def test_convert_linear_to_db(linear: float, db: float) -> None:
    assert math.isclose(
        convert_linear_to_db(linear), db, rel_tol=TOLERANCE, abs_tol=TOLERANCE
    )
