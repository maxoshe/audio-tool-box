import inspect
from os import PathLike
from typing import Any, Callable, List, Literal, Type, TypeVar, get_origin, get_args
import click


PARAMETERS_TO_IGNORE = ["self"]

F = TypeVar("F", bound=Callable[..., Any])


def _get_parameter_type(parameter: inspect.Parameter) -> Type:
    if parameter.annotation is inspect.Parameter.empty:
        return str
    return parameter.annotation


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

    return click.option(option_name, type=parameter_type, default=parameter.default)


def get_click_decorators_from_method(method: Callable) -> List[Callable[[F], F]]:
    decorators = []
    for name, parameter in inspect.signature(method).parameters.items():
        if name in PARAMETERS_TO_IGNORE:
            continue
        decorators.append(_get_click_option(parameter))
    return decorators
