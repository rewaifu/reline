import orjson
import typer
from . import Pipeline


def main(config: str = typer.Option('config.json', '--config', '-c', help='Path to the configuration JSON file')):
    try:
        with open(config, encoding='utf-8') as f:
            data = f.read()
            data = orjson.loads(data)

        Pipeline.from_json(data).process_linear()
    except FileNotFoundError:
        typer.echo(f"Error: Config file '{config}' not found.")
        raise typer.Exit(1)
    except orjson.JSONDecodeError:
        typer.echo(f"Error: Invalid JSON in config file '{config}'")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f'Error during processing: {str(e)}')
        raise typer.Exit(1)


def cli():
    """Entry point for the CLI."""
    typer.run(main)


if __name__ == '__main__':
    cli()
