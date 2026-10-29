"""`spar eval` subcommands: seal, verify and cutoff (novelty and report come later)."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Metrics, statistics, pre-registration.", no_args_is_help=True)

POLICY = Path("configs/eval/prereg.yaml")
PolicyOption = Annotated[Path, typer.Option(help="The cutoff policy file.")]
PreregOption = Annotated[
    Path | None, typer.Option(help="Where manifests live. Defaults to $SPAR_DATA/prereg.")
]


@app.command()
def seal(
    candidates: Annotated[
        Path, typer.Argument(help="Ranked candidates, a .json or .parquet file.")
    ],
    threshold: Annotated[float, typer.Option(help="The coder threshold from the coder study.")],
    policy: PolicyOption = POLICY,
    prereg_dir: PreregOption = None,
    name: Annotated[str | None, typer.Option(help="Defaults to <scene>-<UTC time>.")] = None,
    tag: Annotated[bool, typer.Option(help="Make the git tag prereg/<name>.")] = True,
    sign: Annotated[bool, typer.Option(help="Sign the git tag.")] = True,
) -> None:
    """Freeze the strategy code, threshold and ranked candidates into a sealed manifest."""
    from spar_eval.prereg.seal import SealError
    from spar_eval.prereg.seal import seal as seal_manifest
    from spar_schema import load_scene
    from spar_schema.prereg import CutoffPolicy, default_prereg_dir

    cutoff = CutoffPolicy.load(policy)
    try:
        sealed = seal_manifest(
            candidates,
            scene=load_scene(Path("scenes") / cutoff.scene),
            threshold=threshold,
            policy=cutoff,
            prereg_dir=prereg_dir or default_prereg_dir(),
            name=name,
            tag=tag,
            sign=sign,
        )
    except SealError as exc:
        typer.echo(f"Not sealed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Sealed {sealed.manifest}\nsha256 {sealed.sha256}")
    if sealed.tag:
        typer.echo(
            f"Tagged {sealed.tag}. To make the seal public, run: git push origin {sealed.tag}"
        )
    else:
        typer.echo("No git tag was made, so nothing outside this machine records the seal yet.")
    typer.echo(f"For a public timestamp, run: ots stamp {sealed.manifest}")


@app.command()
def verify(
    manifest: Annotated[Path, typer.Argument(help="A .manifest.json file.")],
    repo: Annotated[Path, typer.Option(help="The git repo holding the tag.")] = Path("."),
) -> None:
    """Check a sealed manifest against its hash, its git tag and its signature."""
    from spar_eval.prereg.verify import verify as verify_manifest
    from spar_schema.prereg import ManifestError

    try:
        result = verify_manifest(manifest, repo=repo)
    except ManifestError as exc:
        typer.echo(f"Not valid: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Hash matches: sha256 {result.sha256}")
    typer.echo(f"Git tag: {result.tag}. Signature: {result.signature}.")
    typer.echo(f"Timestamp proof: {'present' if result.timestamp_proof else 'none yet'}.")
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def cutoff(policy: PolicyOption = POLICY, prereg_dir: PreregOption = None) -> None:
    """Show the cutoff date and whether footage after it may be processed."""
    from spar_schema.prereg import CutoffPolicy, default_prereg_dir, sealed_manifest

    rules = CutoffPolicy.load(policy)
    folder = prereg_dir or default_prereg_dir()
    sealed = sealed_manifest(folder, rules.scene)
    typer.echo(f"Cutoff: {rules.cutoff} for {rules.scene}.")
    if sealed:
        typer.echo(f"Sealed: {sealed.name}. Footage after the cutoff may be processed.")
    else:
        typer.echo("Not sealed. Footage after the cutoff can be downloaded but not processed.")
