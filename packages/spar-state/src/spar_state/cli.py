"""`spar state` subcommands: run."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Task state, score check, events, strategy codes.", no_args_is_help=True)


@app.callback()
def _group() -> None:
    # Keeps `run` a subcommand while it is the only one; Typer would otherwise collapse it.
    pass


@app.command()
def run(
    episode: Annotated[Path, typer.Argument(help="An episode folder.")],
    scene: Annotated[
        Path | None,
        typer.Option(help="The scene folder. Defaults to scenes/<the episode's scene>."),
    ] = None,
    tolerance: Annotated[float, typer.Option(help="Points the final score may be off by.")] = 0.0,
    allow_incomplete_scene: Annotated[
        bool, typer.Option(help="Write even if the scene has TODO values. For development only.")
    ] = False,
) -> None:
    """Score check, events and strategy codes for one episode, written back into it."""
    from spar_schema import EpisodeReader, load_scene
    from spar_state.run import run_episode

    meta = EpisodeReader(episode).meta
    loaded = load_scene(scene or Path("scenes") / meta.scene, rules_version=meta.rules_version)
    result = run_episode(
        episode, loaded, tolerance=tolerance, allow_incomplete_scene=allow_incomplete_scene
    )
    for checkpoint, check in result.checks.items():
        verdict = {True: "pass", False: "FAIL", None: "can't compare"}[check.passed]
        if check.official is not None:
            detail = f"computed {check.computed}, official {check.official}"
        else:
            detail = f"computed winner {check.computed_winner}, official {check.official_winner}"
        typer.echo(f"{checkpoint}: {verdict} ({detail})")
    typer.echo(f"events: {result.events} written")
    if result.codes is None:
        typer.echo("strategy codes: none, the scene has no strategy plugin")
    else:
        typer.echo(f"strategy codes: written for {', '.join(result.codes)}")
