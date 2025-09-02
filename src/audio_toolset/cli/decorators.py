import inspect
from os import PathLike
from typing import Callable, Literal, get_origin, get_args

import click

PARAMETERS_TO_IGNORE = ["self"]


def get_parameter_type(parameter: inspect.Parameter) -> type:
    if parameter.annotation is inspect.Parameter.empty:
        return str
    return parameter.annotation


def format_as_cli_option(parameter_name: str) -> str:
    return f"--{parameter_name.replace('_', '-')}"


def get_click_option(parameter: inspect.Parameter) -> click.Option:
    parameter_type = get_parameter_type(parameter)
    option_name = format_as_cli_option(parameter.name)

    if parameter_type is bool:
        return click.option(option_name, is_flag=True, default=parameter.default)

    if get_origin(parameter_type) is Literal:
        return click.option(
            option_name,
            type=click.Choice(get_args(parameter_type)),
            default=parameter.default,
        )

    if get_origin(parameter_type) is PathLike:
        return click.option(
            option_name,
            type=click.Path(),
            default=parameter.default,
        )

    return click.option(option_name, type=parameter_type, default=parameter.default)


def get_click_decorators_from_method(method: Callable):
    decorators = []
    for name, parameter in inspect.signature(method).parameters.items():
        if name in PARAMETERS_TO_IGNORE:
            continue
        decorators.append(get_click_option(parameter))

    def decorator(func: Callable) -> Callable:
        for decorator in reversed(decorators):
            func = decorator(func)
        return func

    return decorator
