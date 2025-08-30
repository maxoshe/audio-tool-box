import numpy as np

from audio_toolset.constants.math import LOG_FLOOR


def convert_db_to_factor(db: np.ndarray | float) -> np.ndarray | float:
    return 10 ** (db / 20)


def convert_power_to_db(value: np.ndarray | float) -> np.ndarray | float:
    return 10 * np.log10(value + LOG_FLOOR)


def convert_linear_to_db(value: np.ndarray | float) -> np.ndarray | float:
    return 20 * np.log10(np.abs(value) + LOG_FLOOR)
