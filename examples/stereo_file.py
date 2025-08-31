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
