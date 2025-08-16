import numpy as np
import soundfile as sf
from pydantic import BaseModel
from soundfile import _SoundFileInfo
from typing import List, Optional
from os import PathLike


class AudioDataError(Exception): ...


class StereoJoinError(Exception): ...


CHANNEL_AXIS = 1


class AudioData(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    info: Optional[_SoundFileInfo] = None
    sample_rate: int
    data: np.ndarray

    @classmethod
    def read_from_file(cls, file_path: PathLike) -> "AudioData":
        info = sf.info(file_path)
        data, sample_rate = sf.read(file_path, always_2d=False)
        return cls(info=info, data=data, sample_rate=sample_rate)

    def write_to_file(self, output_path: PathLike) -> None:
        sf.write(output_path, self.data, self.sample_rate)

    def is_mono(self) -> bool:
        return self.data.ndim == 1

    def get_number_of_samples(self) -> int:
        return len(self.data)

    def get_number_of_channels(self) -> int:
        if self.is_mono():
            return 1
        return self.data.shape[CHANNEL_AXIS]

    def split_to_mono(self) -> List["AudioData"]:
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
