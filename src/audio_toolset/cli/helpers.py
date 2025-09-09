import inspect
import re
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

ANY_WHITESPACE_REGEX = r"\s+"
METHOD_DESCRIPTION_REGEX = r"(.*)\s*Args:"
ARGUMENT_DOCSTRING_PATTERN = r"{argument_name}[\s*]\(.*\):(.*)\n"

F = TypeVar("F", bound=Callable[..., Any])


class MethodInspectionError(Exception): ...


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
    raise MethodInspectionError(f"Failed unwrapping type from {parameter_type}")


def _format_as_cli_option(parameter_name: str) -> str:
    return f"--{parameter_name.replace('_', '-')}"


def _get_click_option(parameter: inspect.Parameter, help: str) -> Callable[[F], F]:
    parameter_type = _get_parameter_type(parameter)
    option_name = _format_as_cli_option(parameter.name)

    if parameter_type is bool:
        return click.option(
            option_name, is_flag=True, default=parameter.default, help=help
        )

    if get_origin(parameter_type) is Literal:
        return click.option(
            option_name,
            type=click.Choice(get_args(parameter_type)),
            default=parameter.default,
            help=help,
        )

    if get_origin(parameter_type) is PathLike:
        return click.option(
            option_name, type=click.Path(), default=parameter.default, help=help
        )

    if _parameter_is_optional(parameter):
        return click.option(
            option_name,
            type=_unwrap_type_from_optional_parameter(parameter),
            default=parameter.default,
            help=help,
        )

    return click.option(
        option_name, type=parameter_type, default=parameter.default, help=help
    )


def _get_docstring_from_method(method: Callable) -> str:
    docstring = inspect.getdoc(method)
    if not docstring:
        raise MethodInspectionError("User facing method is missing docstring")
    return docstring


def _get_parameter_help_from_docstring(docstring: str, argument_name: str) -> str:
    pattern = ARGUMENT_DOCSTRING_PATTERN.format(argument_name=argument_name)
    match = re.search(pattern, docstring)
    if not match:
        raise MethodInspectionError(f"Missing argument {argument_name} in {docstring}")
    return re.sub(ANY_WHITESPACE_REGEX, " ", match.group(1).strip())


def get_click_decorators_from_method(method: Callable) -> list[Callable[[F], F]]:
    decorators = []
    docstring = _get_docstring_from_method(method)
    for name, parameter in inspect.signature(method).parameters.items():
        if name in PARAMETERS_TO_IGNORE:
            continue
        help = _get_parameter_help_from_docstring(
            docstring=docstring,
            argument_name=parameter.name,
        )
        decorators.append(_get_click_option(parameter, help))
    return decorators


def generate_command_help_from_method(method: Callable) -> str:
    docstring = _get_docstring_from_method(method)
    match = re.search(METHOD_DESCRIPTION_REGEX, docstring)
    if not match:
        raise MethodInspectionError(f"Missing docstring for {method}")
    return re.sub(ANY_WHITESPACE_REGEX, " ", match.group(1).strip())
