from audio_toolset.channel import Channel

track = Channel("my_audio.wav")

track.normalize(target_db=-10)  # Normalize to safe headroom
track.compressor(threshold_db=-20, compression_ratio=4)  # Smooth dynamic range

track.write("audio_dynamics_fixed.wav")
