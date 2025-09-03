from math import floor
from os import PathLike

import numpy as np
import soundfile as sf
from pydantic import BaseModel
from soundfile import _SoundFileInfo


class AudioDataError(Exception): ...


class StereoJoinError(Exception): ...


CHANNEL_AXIS = 1


class AudioData(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    info: _SoundFileInfo | None = None
    sample_rate: int
    data: np.ndarray

    @classmethod
    def read_from_file(cls, file_path: PathLike[str]) -> "AudioData":
        info = sf.info(file_path)
        data, sample_rate = sf.read(file_path, always_2d=False)
        return cls(info=info, data=data, sample_rate=sample_rate)

    def write_to_file(self, output_path: PathLike[str]) -> None:
        sf.write(output_path, self.data, self.sample_rate)

    def is_mono(self) -> bool:
        return self.data.ndim == 1

    def get_number_of_samples(self) -> int:
        return len(self.data)

    def get_number_of_channels(self) -> int:
        if self.is_mono():
            return 1
        return self.data.shape[CHANNEL_AXIS]

    def get_sample_period_s(self) -> float:
        return 1 / self.sample_rate

    def get_duration_s(self) -> float:
        return self.get_sample_period_s() * self.get_number_of_samples()

    def get_peak(self) -> float:
        return np.max(np.abs(self.data))

    def get_nyquist_frequency(self) -> int:
        return floor(self.sample_rate / 2)

    def split_to_mono(self) -> list["AudioData"]:
        if self.is_mono():
            raise AudioDataError("Can't split mono audio")
        return [
            AudioData(sample_rate=self.sample_rate, data=self.data[:, channel])
            for channel in range(self.get_number_of_channels())
        ]

    def sum_to_mono(self) -> "AudioData":
        if self.is_mono():
            raise AudioDataError("Can't sum mono audio")
        return AudioData(
            sample_rate=self.sample_rate, data=np.mean(self.data, axis=CHANNEL_AXIS)
        )

    def get_copy(self) -> "AudioData":
        return AudioData(
            info=self.info, sample_rate=self.sample_rate, data=np.copy(self.data)
        )


def join_to_stereo(
    left_channel: AudioData,
    right_channel: AudioData,
) -> AudioData:
    if not (left_channel.is_mono() and right_channel.is_mono()):
        raise StereoJoinError("Both channels must be mono")
    if left_channel.sample_rate != right_channel.sample_rate:
        raise StereoJoinError("Both channels must have the same sample rate")

    return AudioData(
        sample_rate=left_channel.sample_rate,
        data=np.column_stack((left_channel.data, right_channel.data)),
    )
