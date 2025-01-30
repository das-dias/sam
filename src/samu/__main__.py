__doc__ = """Usage: samu.py [-vgrh] FILE

Arguments:
  FILE  input file

Options:
  -h --help
  -v  verbose mode
  -r  extract resistance
  -g  gui mode

"""
from .samu import cli

if __name__ == "__main__":
    cli()