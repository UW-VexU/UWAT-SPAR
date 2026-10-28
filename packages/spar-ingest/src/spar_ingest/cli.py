"""`spar ingest` subcommands: events, matches, streams, download, cut, count."""

import typer

app = typer.Typer(help="Match records, stream download, cutting.", no_args_is_help=True)
