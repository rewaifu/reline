import re

import orjson

import typer

from reline import Pipeline


def strip_jsonc(jsonc_str):
    jsonc_str = re.sub(r'//.*', '', jsonc_str)
    jsonc_str = re.sub(r'/\*.*?\*/', '', jsonc_str, flags=re.DOTALL)
    return jsonc_str


def main(config_path: str = 'config.json'):
    with open(config_path, encoding='utf-8') as f:
        data = f.read()
        data = strip_jsonc(data)
        data = orjson.loads(data)

    Pipeline.from_json(data).process()


if __name__ == '__main__':
    typer.run(main)
