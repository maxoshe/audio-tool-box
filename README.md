# Audio Toolset

[![CI Workflow status](https://github.com/maxoshe/audio-toolset/actions/workflows/ci.yml/badge.svg)](https://github.com/maxoshe/audio-toolset/actions/workflows/ci.yml)
[![CD Workflow status](https://github.com/maxoshe/audio-toolset/actions/workflows/cd.yml/badge.svg)](https://github.com/maxoshe/audio-toolset/actions/workflows/cd.yml)
[![GitHub Release](https://img.shields.io/github/v/release/maxoshe/audio-toolset)](https://github.com/maxoshe/audio-toolset/releases)
[![PyPI - Version](https://img.shields.io/pypi/v/audio-toolset)](https://pypi.org/project/audio-toolset/)
[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fmaxoshe%2Faudio-toolset%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/maxoshe/audio-toolset/blob/main/pyproject.toml)



Python library and command line interface for **processing audio signals**. 

`audio_toolset` provides an intuitive and flexible way to process audio files in Python. Each `Channel` object handles a single **mono audio file**, allowing you to apply filters, EQ, dynamic processing, gain adjustments, noise reduction, and more.

> [!TIP]
> **Stereo files** can be handled by first splitting them into mono channels using `split_to_mono` and then recombining them with `join_to_stereo`. [see example below](#working-with-stereo-files)

## Installation

```bash
pip install audio-toolset
```

## Examples

### Quickstart

```python
from audio_toolset.channel import Channel

track = Channel("my_audio.wav")

track.lowpass(cutoff_frequency=12000)  # Remove harsh highs
track.highpass(cutoff_frequency=80)  # Remove low-end rumble
track.normalize(target_db=-1)  # Normalize

track.write("my_audio_processed.wav")

```

### Command line interface

All methods in `Channel` are exposed as subcommands in the `audio-toolset` CLI and can be chained in any order.
this command will produce an identical result to the python script above
```bash
audio-toolset --source my_audio.wav \
  lowpass --cutoff-frequency 12000 \
  highpass --cutoff-frequency 80 \
  normalize --target-db -1 \
  write --output-path my_audio_processed.wav
```

### Process multiple files
This is very handy when you need to apply the same processing to multiple files, for example a directory with sound effects.

```python
import pathlib

from audio_toolset.channel import Channel

input_dir = pathlib.Path("input_audio")
output_dir = pathlib.Path("cleaned_audio")
output_dir.mkdir(exist_ok=True)

for wav_file in input_dir.glob("*.wav"):
    track = Channel(wav_file)

    track.lowpass(cutoff_frequency=12000)  # Remove harsh highs
    track.highpass(cutoff_frequency=80)  # Remove low-end rumble
    track.normalize(target_db=-1)  # Normalize

    track.write(output_dir / wav_file.name)

```

### Working with stereo files
```python
from audio_toolset.audio_data import AudioData, join_to_stereo
from audio_toolset.channel import Channel

# Load stereo audio file
stereo_data = AudioData.read_from_file("my_audio.wav")

# Split stereo into left and right channels
left_data, right_data = stereo_data.split_to_mono()
left_channel = Channel(left_data)
right_channel = Channel(right_data)

# Process each channel
for track in [left_channel, right_channel]:
    track.lowpass(cutoff_frequency=12000)  # Remove harsh highs
    track.highpass(cutoff_frequency=80)  # Remove low-end rumble
    track.normalize(target_db=-1)  # Normalize

# Combine channels back to stereo and save
join_to_stereo(left_channel.audio_data, right_channel.audio_data).write_to_file(
    "my_audio_processed.wav"
)

```

## Features

- High level [`Channel`](https://github.com/maxoshe/audio-toolset/blob/main/src/audio_toolset/channel.py) object for simple user interface
- Load audio from a file or [`AudioData`](https://github.com/maxoshe/audio-toolset/blob/a609df16c37a2e08653ee153a3ecde7f48faa41c/src/audio_toolset/audio_data.py)
- Gain processing
  - `gain`, `normalize`, `fade`
- Filters
  - `lowpass`, `highpass`, `eq_band`
  - `noise_reduction` (spectral gating)
- Dynamics
  - `compressor`, `limiter`, `soft_clipping`
- Plotting tools ([see examples](#plotting))
  - `plot_signal` - generate signal time and frequency plot
  - Dynamic plots - `compressor`, `limiter` can generate attenuation plots
  - Bode plots - `lowpass`, `highpass`, `eq_band` can generate filter bode plots

## Plotting

`audio_toolset` provides multiple plotting tools to visualize signals, filters, and dynamics. All plots are generated interactively using [**Plotly**](https://plotly.com) and can be saved as images from the interactive viewer.

### Signal Plots

Use `plot_signal()` to visualize the time-domain and frequency-domain representation of an audio channel.

[<img width="1674" alt="signal-plot" src="https://github.com/user-attachments/assets/c993c26c-4001-4028-82c3-b6419848ba5f" />](https://github.com/user-attachments/assets/c993c26c-4001-4028-82c3-b6419848ba5f)


### 2. Filter (Bode) Plots

Filter plots show the frequency response (Bode plot) of filters applied using `eq_band(plot_filter_bode=True)`, `highpass(plot_filter_bode=True)`, or `lowpass(plot_filter_bode=True)`.

[<img width="1674" alt="lowpass-plot" src="https://github.com/user-attachments/assets/53f7c585-34d3-4485-b624-682968d9cab0" />](https://github.com/user-attachments/assets/53f7c585-34d3-4485-b624-682968d9cab0)

[<img width="1674" alt="parametric-eq-plot" src="https://github.com/user-attachments/assets/b524addc-5e0e-4b44-a0a6-22e68e588924" />](https://github.com/user-attachments/assets/b524addc-5e0e-4b44-a0a6-22e68e588924)


### 3. Dynamics Plots

Dynamics plots visualize the response of compressors and limiters applied to a channel. Generated by `compressor(plot_compressor_response=True)` or `limiter(plot_limiter_response=True)`

[<img width="1674" alt="compressor-plot" src="https://github.com/user-attachments/assets/a7db63af-e113-47a0-a536-d2e27a348a1d" />](https://github.com/user-attachments/assets/a7db63af-e113-47a0-a536-d2e27a348a1d)
