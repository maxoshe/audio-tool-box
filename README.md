# AudioToolBox
A python module for processing audio.

AudioToolBox provides a user friendly way to process audio in python.
Each channel object is used to import, process and export a single mono wav file.

Dependencies
--
numpy, scipy, soundfile, matplotlib

Example
--
Process a guitar signal using a channel strip

    import AudioToolBox
    guitar = AudioToolBox.channel('guitar.wav')
    guitar.highpass(fc=100, db_per_octave=6)
    guitar.eq_band(fc=2000, gain_db=3, q=1)
    guitar.lowpass(fc=10000, db_per_octave=12)
    guitar.normalize(target_db=-1)
    guitar.compressor(tresh_db=-20, ratio=4, attack_ms=50, release_ms=100)
    guitar.export('guitar_processed.wav')
