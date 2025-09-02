from typing import Callable
import click

from audio_toolset.channel import Channel
from audio_toolset.cli.helpers import get_click_decorators_from_method, F


pass_channel = click.make_pass_decorator(Channel, ensure=True)


def click_audio_toolset_options(method: Callable) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        for dec in reversed(get_click_decorators_from_method(method)):
            func = dec(func)
        return func

    return decorator
