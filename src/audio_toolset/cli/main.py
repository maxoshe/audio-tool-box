import click
from audio_toolset.channel import Channel
from audio_toolset.cli.decorators import get_click_decorators_from_method

pass_channel = click.make_pass_decorator(Channel, ensure=True)


@click.group(chain=True)
@click.option("--source", type=click.Path(), help="Path to an audio file")
@click.pass_context
def audio_toolset_cli(context: click.Context, source):
    context.obj = Channel(source=source)


@audio_toolset_cli.command()
@pass_channel
@get_click_decorators_from_method(Channel.lowpass)
def lowpass(channel: Channel, **kwargs):
    channel.lowpass(**kwargs)


@audio_toolset_cli.command()
@pass_channel
@get_click_decorators_from_method(Channel.write)
def write(channel: Channel, **kwargs):
    channel.write(**kwargs)


if __name__ == "__main__":
    audio_toolset_cli()
