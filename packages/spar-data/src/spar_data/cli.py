"""`spar data` subcommands: catalog and start-states (convert and publish come later)."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Dataset catalog and converters.", no_args_is_help=True)

RootOption = Annotated[
    Path | None, typer.Option(help="Where episodes live. Defaults to $SPAR_DATA/episodes.")
]


@app.command()
def catalog(
    root: RootOption = None,
    scene: Annotated[str | None, typer.Option(help="Only this scene.")] = None,
    out: Annotated[Path | None, typer.Option(help="Also save the index as a Parquet file.")] = None,
) -> None:
    """Count the episodes by scene, rules version, program and score check."""
    import pyarrow.parquet as pq

    from spar_data.catalog import default_episodes_dir, scan

    found = scan(root or default_episodes_dir())
    if scene:
        found = found.filter(scene=scene)
    typer.echo(f"{len(found)} episodes")
    for (name, rules, program, check), n in sorted(
        found.counts("scene", "rules_version", "program", "score_check").items()
    ):
        typer.echo(f"  {name} {rules} {program}: {n} {check}")
    for folder, reason in found.skipped:
        typer.echo(f"  skipped {folder}: {reason}")
    if out:
        pq.write_table(found.table(), out)
        typer.echo(f"wrote {out}")


@app.command("start-states")
def start_states(
    scene: Annotated[str, typer.Option(help="The scene, as named in scenes/.")],
    n: Annotated[int, typer.Option(help="How many moments to sample.")] = 10000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    passed_only: Annotated[
        bool, typer.Option(help="Only episodes that passed the score check.")
    ] = True,
    root: RootOption = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Defaults to $SPAR_DATA/start_states/<scene>-<rules version>.parquet."),
    ] = None,
) -> None:
    """Sample moments from real episodes into start_states.parquet."""
    from spar_data.catalog import default_episodes_dir, scan
    from spar_data.start_states import sample_start_states
    from spar_schema import load_scene, write_start_states

    loaded = load_scene(Path("scenes") / scene)
    found = scan(root or default_episodes_dir()).filter(
        scene=loaded.name,
        rules_version=loaded.rules_version,
        score_check="passed" if passed_only else None,
    )
    if not len(found):
        typer.echo("No episodes to sample from.", err=True)
        raise typer.Exit(1)
    states = sample_start_states(found, loaded, n, seed=seed)
    path = out or (
        default_episodes_dir().parent
        / "start_states"
        / f"{loaded.name}-{loaded.rules_version}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_start_states(path, states, loaded)
    typer.echo(f"wrote {len(states)} start states from {len(found)} episodes to {path}")
