import numpy as np


def convert_db_to_factor(db: float) -> float:
    return 10 ** (db / 20)


def convert_power_to_db(value: np.ndarray | float) -> np.ndarray | float:
    return 10 * np.log10(value + 1e-12)


def convert_linear_to_db(value: np.ndarray | float) -> np.ndarray | float:
    return 20 * np.log10(np.abs(value) + 1e-12)
