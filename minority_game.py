"""Run: python minority_game.py --help"""
import argparse
import json
from pathlib import Path
from GameSimulator.krillion.minority import Config, simulate, write_report


def main():
    parser = argparse.ArgumentParser(description='Minority answer game with private histories')
    for name, default in vars(Config()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    parser.add_argument('--output', default='minority_game.html')
    parser.add_argument('--json-output', default='minority_game.json')
    args = vars(parser.parse_args())
    output, json_output = args.pop('output'), args.pop('json_output')
    try:
        result = simulate(Config(**args))
    except ValueError as error:
        parser.error(str(error))
    write_report(result, output)
    Path(json_output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Report: {Path(output).resolve()}')
    print(f'Full results: {Path(json_output).resolve()}')


if __name__ == '__main__':
    main()
