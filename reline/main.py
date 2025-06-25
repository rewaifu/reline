import traceback
from typing import Annotated, Optional
import orjson
import typer
from . import Pipeline, __version__
from rich import print
from rich.table import Table

app = typer.Typer(pretty_exceptions_show_locals=False)


def version_callback(value: bool):
    if not value:
        return

    table = Table()
    table.add_column('Dependency', style='cyan', no_wrap=True)
    table.add_column('Version', style='magenta')
    table.add_column('Details', style='green')

    table.add_row('reline', __version__ or 'unknown', '-')

    import resselt

    table.add_row('resselt', getattr(resselt, '__version__', 'unknown'), '-')

    import torch

    cuda_status = 'Yes' if torch.cuda.is_available() else 'No'
    table.add_row('torch', torch.__version__, f'CUDA Available: {cuda_status}')

    print(table)
    raise typer.Exit()


@app.command()
def main(
    config: str = typer.Option('config.json', '--config', '-c', help='Path to the configuration JSON file'),
    version: Annotated[Optional[bool], typer.Option('--version', callback=version_callback)] = None,
):
    try:
        with open(config, encoding='utf-8') as f:
            data = f.read()
            data = orjson.loads(data)

        Pipeline.from_json(data).process_linear()
    except orjson.JSONDecodeError:
        typer.echo(f"Error: Invalid JSON in config file '{config}': {traceback.format_exc()}")
        raise typer.Exit(1)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == '__main__':
    cli()
