"""`spar perception` subcommands: autolabel, review, train-detector, run."""

import typer

app = typer.Typer(
    help="Auto-labeling, detector training, tracking, field calibration.", no_args_is_help=True
)
