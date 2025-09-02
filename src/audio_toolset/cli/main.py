from os import PathLike
import click
from audio_toolset.channel import Channel
from audio_toolset.cli import decorators


@click.group(chain=True)
@click.option(
    "--source",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to an audio file",
)
@click.pass_context
def audio_toolset_cli(context: click.Context, source: PathLike[str]) -> None:
    context.obj = Channel(source=source)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.write)
def write(channel: Channel, **kwargs) -> None:
    channel.write(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.plot_signal)
def plot_signal(channel: Channel, **kwargs) -> None:
    channel.plot_signal(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.gain)
def gain(channel: Channel, **kwargs) -> None:
    channel.gain(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.normalize)
def normalize(channel: Channel, **kwargs) -> None:
    channel.normalize(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.fade)
def fade(channel: Channel, **kwargs) -> None:
    channel.fade(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.lowpass)
def lowpass(channel: Channel, **kwargs) -> None:
    channel.lowpass(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.highpass)
def highpass(channel: Channel, **kwargs) -> None:
    channel.highpass(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.eq_band)
def eq_band(channel: Channel, **kwargs) -> None:
    channel.eq_band(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.noise_reduction)
def noise_reduction(channel: Channel, **kwargs) -> None:
    channel.noise_reduction(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.compressor)
def compressor(channel: Channel, **kwargs) -> None:
    channel.compressor(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.limiter)
def limiter(channel: Channel, **kwargs) -> None:
    channel.limiter(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.soft_clipping)
def soft_clipping(channel: Channel, **kwargs) -> None:
    channel.soft_clipping(**kwargs)


if __name__ == "__main__":
    audio_toolset_cli()
