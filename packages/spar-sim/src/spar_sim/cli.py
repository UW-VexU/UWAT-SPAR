"""`spar sim` subcommands: fit, replay and bench."""

import os
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Plan-level simulator (JAX).", no_args_is_help=True)

SceneOption = Annotated[str, typer.Option(help="The scene, as named in scenes/.")]
TimingOption = Annotated[
    Path | None,
    typer.Option(help="Timing from spar sim fit. Defaults to $SPAR_DATA/sim/<scene>-<rules>.json."),
]


@app.command()
def fit(
    scene: SceneOption,
    root: Annotated[
        Path | None, typer.Option(help="Where episodes live. Defaults to $SPAR_DATA/episodes.")
    ] = None,
    min_samples: Annotated[int, typer.Option(help="Fewest examples to fit an action.")] = 5,
    out: TimingOption = None,
) -> None:
    """Fit action durations, failure rates, robot speeds and the detour ratio from episodes with
    actions."""
    from spar_sim.fit import fit_timing

    loaded = _scene(scene)
    episodes = [e for e in _episodes(root or _data() / "episodes", loaded) if e.actions is not None]
    if not episodes:
        typer.echo("No episodes with actions.parquet; run spar state run first.", err=True)
        raise typer.Exit(1)
    timing = fit_timing(episodes, loaded, min_samples=min_samples)
    path = out or _timing_path(loaded)
    path.parent.mkdir(parents=True, exist_ok=True)
    timing.save(path)
    typer.echo(
        f"fitted from {timing.episodes} episodes: robots drive at {timing.speed_mps:.2f} m/s, "
        f"on paths {timing.detour:.2f} times the straight line"
    )
    for name, row in timing.actions.items():
        fitted = f"{row.n} examples" if row.n else "default"
        typer.echo(f"  {name}: {row.median_s:.2f} s median, p_fail {row.p_fail:.2f} ({fitted})")
    typer.echo(f"wrote {path}")


@app.command()
def replay(
    episodes: Annotated[Path, typer.Argument(help="An episode folder, or a folder of them.")],
    scene: SceneOption,
    timing: TimingOption = None,
    sampled: Annotated[
        bool, typer.Option(help="Draw durations and failures from the timing, not the log.")
    ] = False,
    svg: Annotated[Path | None, typer.Option(help="Draw one episode's replay here.")] = None,
) -> None:
    """Replay episodes' actions from their start and report the score error."""
    from spar_sim.fit import Timing
    from spar_sim.replay import replay_episode, replay_summary
    from spar_sim.viz import save_replay_svg

    loaded = _scene(scene)
    fitted = None
    if timing or sampled:
        fitted = Timing.load(timing or _timing_path(loaded))
    found = [e for e in _episodes(episodes, loaded) if e.actions is not None]
    if not found:
        typer.echo("No episodes with actions.parquet to replay.", err=True)
        raise typer.Exit(1)
    results = []
    for episode in found:
        result = replay_episode(
            episode.path, loaded, fitted, logged_times=not sampled, trace=svg is not None
        )
        results.append(result)
        typer.echo(
            f"{result.episode_id}: simulated {result.final}, official {result.official}, "
            f"computed {result.computed}"
        )
    if svg:
        save_replay_svg(results[0].env, results[0].states, svg, title=results[0].episode_id)
        typer.echo(f"wrote {svg}")
    summary = replay_summary(results)
    typer.echo(", ".join(f"{k} {v:.3g}" for k, v in summary.items()))


@app.command()
def bench(
    scene: SceneOption,
    program: Annotated[str, typer.Option(help="The program to simulate.")],
    n_envs: Annotated[int, typer.Option(help="Matches run at once.")] = 1024,
    steps: Annotated[int, typer.Option(help="Steps per match.")] = 200,
    timing: TimingOption = None,
) -> None:
    """Simulator throughput, in steps per second, with random actions."""
    from spar_sim.bench import benchmark
    from spar_sim.env import PlanEnv
    from spar_sim.fit import Timing

    loaded = _scene(scene)
    fitted = Timing.load(timing) if timing else None
    result = benchmark(PlanEnv(loaded, fitted, program=program), n_envs=n_envs, n_steps=steps)
    typer.echo(
        f"{result['env_steps_per_s']:,.0f} steps/s on {result['device']} "
        f"({result['envs']} matches x {result['steps']} steps in {result['seconds']:.2f} s)"
    )


def _scene(name: str):
    from spar_schema import load_scene

    return load_scene(Path("scenes") / name)


def _data() -> Path:
    root = os.environ.get("SPAR_DATA")
    if not root:
        raise typer.BadParameter("SPAR_DATA is not set; point it at the data root")
    return Path(root)


def _timing_path(scene) -> Path:
    return _data() / "sim" / f"{scene.name}-{scene.rules_version}.json"


def _episodes(root: Path, scene):
    """The episodes of this scene and rules version at or under root."""
    from spar_schema import EpisodeReader

    folders = (
        [root]
        if (root / "episode.json").is_file()
        else sorted(p.parent for p in root.rglob("episode.json"))
    )
    found = []
    for folder in folders:
        episode = EpisodeReader(folder)
        if episode.meta.scene == scene.name and episode.meta.rules_version == scene.rules_version:
            found.append(episode)
    return found
