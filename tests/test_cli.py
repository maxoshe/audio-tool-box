import inspect
from collections.abc import Callable

import pytest

from audio_toolset.channel import Channel
from audio_toolset.cli.helpers import get_click_decorators_from_method


@pytest.mark.parametrize(
    "method",
    [
        Channel.write,
        Channel.plot_signal,
        Channel.gain,
        Channel.normalize,
        Channel.fade,
        Channel.lowpass,
        Channel.highpass,
        Channel.eq_band,
        Channel.noise_reduction,
        Channel.compressor,
        Channel.limiter,
        Channel.soft_clipping,
    ],
)
def test_cli_decorators_match_parameters(method: Callable) -> None:
    decorators = get_click_decorators_from_method(method)
    num_params = len(inspect.signature(method).parameters) - 1
    assert len(decorators) == num_params
