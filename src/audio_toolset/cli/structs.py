from dataclasses import dataclass

from audio_toolset.channel import Channel


@dataclass
class ContextObject:
    channel: Channel
    debug: bool
