from audio_toolset.channel import Channel

track = Channel("my_audio.wav")

track.lowpass(cutoff_frequency=12000)  # Remove harsh high frequencies
track.highpass(cutoff_frequency=80)  # Remove low-end rumble
track.normalize(target_db=-1)  # Peak normalize

track.write("my_audio_processed.wav")
