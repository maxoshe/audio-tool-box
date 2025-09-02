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
@decorators.click_audio_toolset_options(Channel.lowpass)
def lowpass(channel: Channel, **kwargs):
    channel.lowpass(**kwargs)


@audio_toolset_cli.command()
@decorators.pass_channel
@decorators.click_audio_toolset_options(Channel.write)
def write(channel: Channel, **kwargs):
    channel.write(**kwargs)


if __name__ == "__main__":
    audio_toolset_cli()
