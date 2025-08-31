import pathlib
from audio_toolset.channel import Channel

input_dir = pathlib.Path("input_audio")
output_dir = pathlib.Path("cleaned_audio")
output_dir.mkdir(exist_ok=True)

for wav_file in input_dir.glob("*.wav"):
    track = Channel(wav_file)

    track.fade(fade_duration_ms=10)  # Remove start/end pops if present
    track.lowpass(cutoff_frequency=12000)  # Remove harsh high frequencies
    track.highpass(cutoff_frequency=80)  # Remove low-end rumble
    track.normalize(target_db=-1)  # Normalize

    track.write(output_dir / wav_file.name)
