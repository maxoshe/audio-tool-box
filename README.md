# audio_tool_box
A python library for processing audio signals.

`audio_tool_box` provides a user friendly way to process audio in python.
Each channel object is used to import, process and export a single mono wav file.
The `split_to_mono` and `join_to_stereo` functions can be used to work with stereo files.

Example
--
Process a guitar signal using a channel strip
```python
from audio_tool_box.channel import Channel

guitar = Channel('guitar.wav')
guitar.highpass(fc=100, db_per_octave=6)
guitar.eq_band(fc=2000, gain_db=3, q=1)
guitar.lowpass(fc=10000, db_per_octave=12)
guitar.normalize(target_db=-1)
guitar.compressor(tresh_db=-20, ratio=4, attack_ms=50, release_ms=100)
guitar.export('guitar_processed.wav')
```