from os import PathLike

import click

from audio_toolset.channel import Channel
from audio_toolset.cli import decorators
from audio_toolset.cli.helpers import generate_command_help_from_method
from audio_toolset.cli.structs import ContextObject


@click.group(chain=True)
@click.option(
    "--source",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to an audio file",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Show full tracebacks on error",
)
@click.pass_context
def audio_toolset_cli(
    context: click.Context, source: PathLike[str], debug: bool
) -> None:
    context.obj = ContextObject(channel=Channel(source), debug=debug)


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.write))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.write)
def write(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.write(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"write failed: {e}")


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.plot_signal))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.plot_signal)
def plot_signal(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.plot_signal(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"plot_signal failed: {e}")


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.gain))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.gain)
def gain(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.gain(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"gain failed: {e}")


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.normalize))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.normalize)
def normalize(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.normalize(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"normalize failed: {e}")


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.fade))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.fade)
def fade(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.fade(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"fade failed: {e}")


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.lowpass))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.lowpass)
def lowpass(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.lowpass(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"lowpass failed: {e}")


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.highpass))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.highpass)
def highpass(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.highpass(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"highpass failed: {e}")


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.eq_band))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.eq_band)
def eq_band(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.eq_band(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"eq_band failed: {e}")


@audio_toolset_cli.command(
    help=generate_command_help_from_method(Channel.noise_reduction)
)
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.noise_reduction)
def noise_reduction(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.noise_reduction(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"noise_reduction failed: {e}")


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.compressor))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.compressor)
def compressor(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.compressor(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"compressor failed: {e}")


@audio_toolset_cli.command(help=generate_command_help_from_method(Channel.limiter))
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.limiter)
def limiter(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.limiter(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"limiter failed: {e}")


@audio_toolset_cli.command(
    help=generate_command_help_from_method(Channel.soft_clipping)
)
@decorators.pass_context_object
@decorators.click_audio_toolset_options(Channel.soft_clipping)
def soft_clipping(obj: ContextObject, **kwargs) -> None:
    try:
        obj.channel.soft_clipping(**kwargs)
    except Exception as e:
        if obj.debug:
            raise
        click.echo(f"soft_clipping failed: {e}")


if __name__ == "__main__":
    audio_toolset_cli()
