from collections.abc import Callable

import click

from audio_toolset.cli.helpers import F, get_click_decorators_from_method
from audio_toolset.cli.structs import ContextObject

pass_context_object = click.make_pass_decorator(ContextObject, ensure=True)


def click_audio_toolset_options(method: Callable) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        for dec in reversed(get_click_decorators_from_method(method)):
            func = dec(func)
        return func

    return decorator
