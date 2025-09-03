import inspect
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from audio_toolset.audio_data import AudioData
from audio_toolset.channel import Channel
from audio_toolset.cli.helpers import get_click_decorators_from_method
from audio_toolset.cli.main import audio_toolset_cli


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


def test_cli_commands(runner: CliRunner, test_tone: AudioData, tmp_path: Path):
    file_path = tmp_path / "test.wav"
    test_tone.write_to_file(file_path)

    commands = list(audio_toolset_cli.commands.keys())
    with patch("plotly.graph_objects.Figure.show"):
        for cmd in commands:
            if cmd == "write":
                continue
            result = runner.invoke(
                audio_toolset_cli,
                ["--source", file_path, cmd],
            )
            assert result.exit_code == 0


def test_cli_write(runner: CliRunner, test_tone: AudioData, tmp_path: Path):
    file_path = tmp_path / "test.wav"
    out_path = tmp_path / "test_out.wav"
    test_tone.write_to_file(file_path)

    result = runner.invoke(
        audio_toolset_cli,
        ["--source", file_path, "write", "--output-path", out_path],
    )
    assert result.exit_code == 0
