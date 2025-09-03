from typing import Callable
import click

from audio_toolset.cli.structs import ContextObject
from audio_toolset.cli.helpers import get_click_decorators_from_method, F

pass_context_object = click.make_pass_decorator(ContextObject, ensure=True)


def click_audio_toolset_options(method: Callable) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        for dec in reversed(get_click_decorators_from_method(method)):
            func = dec(func)
        return func

    return decorator
