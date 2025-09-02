import click
from audio_toolset.channel import Channel
from audio_toolset.cli import decorators


@click.group(chain=True)
@click.option("--source", type=click.Path(), help="Path to an audio file")
@click.pass_context
def audio_toolset_cli(context: click.Context, source):
    context.obj = Channel(source=source)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.write)
def write(channel: Channel, **kwargs):
    channel.write(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.plot_signal)
def plot_signal(channel: Channel, **kwargs):
    channel.plot_signal(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.gain)
def gain(channel: Channel, **kwargs):
    channel.gain(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.normalize)
def normalize(channel: Channel, **kwargs):
    channel.normalize(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.fade)
def fade(channel: Channel, **kwargs):
    channel.fade(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.lowpass)
def lowpass(channel: Channel, **kwargs):
    channel.lowpass(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.highpass)
def highpass(channel: Channel, **kwargs):
    channel.highpass(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.eq_band)
def eq_band(channel: Channel, **kwargs):
    channel.eq_band(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.noise_reduction)
def noise_reduction(channel: Channel, **kwargs):
    channel.noise_reduction(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.compressor)
def compressor(channel: Channel, **kwargs):
    channel.compressor(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.limiter)
def limiter(channel: Channel, **kwargs):
    channel.limiter(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.soft_clipping)
def soft_clipping(channel: Channel, **kwargs):
    channel.soft_clipping(**kwargs)


if __name__ == "__main__":
    audio_toolset_cli()
