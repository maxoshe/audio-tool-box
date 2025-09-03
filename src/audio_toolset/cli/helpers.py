import inspect
from collections.abc import Callable
from os import PathLike
from types import UnionType
from typing import (
    Any,
    Literal,
    TypeVar,
    get_args,
    get_origin,
)

import click

PARAMETERS_TO_IGNORE = ["self"]

F = TypeVar("F", bound=Callable[..., Any])


def _get_parameter_type(parameter: inspect.Parameter) -> type:
    if parameter.annotation is inspect.Parameter.empty:
        return str
    return parameter.annotation


def _parameter_is_optional(parameter: inspect.Parameter) -> bool:
    parameter_type = _get_parameter_type(parameter)
    parameter_args = get_args(parameter_type)
    return (
        get_origin(parameter_type) is UnionType
        and len(parameter_args) == 2
        and type(None) in parameter_args
    )


def _unwrap_type_from_optional_parameter(parameter: inspect.Parameter) -> type:
    parameter_type = _get_parameter_type(parameter)
    types = [arg for arg in get_args(parameter_type) if arg is not type(None)]
    if len(types) == 1:
        return types[0]
    raise Exception(f"Failed unwrapping type from {parameter_type}")


def _format_as_cli_option(parameter_name: str) -> str:
    return f"--{parameter_name.replace('_', '-')}"


def _get_click_option(parameter: inspect.Parameter) -> Callable[[F], F]:
    parameter_type = _get_parameter_type(parameter)
    option_name = _format_as_cli_option(parameter.name)

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

    if _parameter_is_optional(parameter):
        return click.option(
            option_name,
            type=_unwrap_type_from_optional_parameter(parameter),
            default=parameter.default,
        )

    return click.option(option_name, type=parameter_type, default=parameter.default)


def get_click_decorators_from_method(method: Callable) -> list[Callable[[F], F]]:
    decorators = []
    for name, parameter in inspect.signature(method).parameters.items():
        if name in PARAMETERS_TO_IGNORE:
            continue
        decorators.append(_get_click_option(parameter))
    return decorators
