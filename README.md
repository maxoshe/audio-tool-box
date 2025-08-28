# Audio Tool Box


![CI Workflow status](https://github.com/maxoshe/audio-tool-box/actions/workflows/ci.yml/badge.svg)
![GitHub top language](https://img.shields.io/github/languages/top/maxoshe/audio-tool-box)


A Python library for **processing audio signals**.

`audio_tool_box` provides an intuitive and flexible way to process audio files in Python. Each `Channel` object handles a single **mono audio file**, allowing you to apply filters, EQ, dynamics processing, gain adjustments, noise reduction, and more.

Stereo files can be handled by first splitting them into mono channels using `split_to_mono` and then recombining them with `join_to_stereo`.

## Features

- High level [`Channel`](https://github.com/maxoshe/audio-tool-box/blob/main/src/audio_tool_box/channel.py) object for simple user interface
- Load audio from a file or [`AudioData`](https://github.com/maxoshe/audio-tool-box/blob/a609df16c37a2e08653ee153a3ecde7f48faa41c/src/audio_tool_box/audio_data.py)
- Gain processing
  - `gain`, `normalize`, `fade`
- Filters
  - `lowpass`, `highpass`, `eq_band`
  - `noise_reduction` (spectral gating)
- Dynamics
  - `compressor`, `limiter`, `soft_clipping`
- Plotting tools
  - `plot_signal` - generate signal time and frequency plot
  - Dynamic plots - `compressor`, `limiter` can generate attenuation plots
  - Bode plots - `lowpass`, `highpass`, `eq_band` can generate filter bode plots

## Installation

```bash
pip install git+https://github.com/maxoshe/audio-tool-box.git
```

## Examples

### Process with a channel strip

```python
from audio_tool_box.channel import Channel

guitar = Channel("guitar.wav")
guitar.highpass(cutoff_frequency=100, db_per_octave=6)
guitar.eq_band(center_frequency=2000, gain_db=3, q_factor=1)
guitar.lowpass(cutoff_frequency=10000, db_per_octave=12)
guitar.normalize(target_db=-1)
guitar.compressor(threshold_db=-20, compression_ratio=4, attack_ms=50, release_ms=100)
guitar.write("guitar_processed.wav")
```

### Process with a chained channel strip

```python
speech = Channel("speech.wav").highpass(cutoff_frequency=100).lowpass(cutoff_frequency=300)
```
