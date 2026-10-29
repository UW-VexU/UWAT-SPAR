"""Freezes the strategy code definition, distance, threshold and ranked candidates into a manifest,
then hashes, tags and timestamps it.

seal() writes $SPAR_DATA/prereg/<name>.manifest.json and the .sha256 file beside it, after making a
git tag whose message carries the hash (signed by default). If tagging fails, nothing is written.
Making the hash public (pushing the tag, and an OpenTimestamps proof) is left to a person,
because it cannot be undone.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from spar_schema import SCHEMA_VERSION, Scene
from spar_schema.prereg import (
    MANIFEST_SUFFIX,
    CutoffPolicy,
    hash_path,
    manifest_bytes,
    sha256_hex,
)

MANIFEST_VERSION = 1
TAG_PREFIX = "prereg/"


class SealError(ValueError):
    """The inputs cannot be sealed as they are."""


@dataclass(frozen=True)
class Sealed:
    manifest: Path
    sha256: str
    tag: str | None


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Ranked candidates from a .json list or a .parquet table, each with a `rank` and a `code`."""
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".parquet":
        rows = pq.read_table(path).to_pylist()
    else:
        raise SealError(f"{path}: candidates must be a .json or .parquet file")
    for row in rows:
        if isinstance(row.get("code"), str):  # Parquet stores the code as JSON text
            row["code"] = json.loads(row["code"])
    return rows


def build_manifest(
    *,
    name: str,
    scene: str,
    rules_version: str,
    strategy_code: dict[str, Any],
    threshold: float,
    candidates: list[dict[str, Any]],
    cutoff: str,
    candidates_sha256: str | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    sealed_at: datetime | None = None,
) -> dict[str, Any]:
    """The manifest's content, after checking that it is complete enough to seal."""
    if _has_todo(strategy_code):
        raise SealError("the strategy code definition still has TODO values")
    if not candidates:
        raise SealError("the candidate list is empty")
    if [c.get("rank") for c in candidates] != list(range(1, len(candidates) + 1)):
        raise SealError("candidates must be ranked 1, 2, 3, ... in order")
    fields = set(strategy_code["fields"])
    for candidate in candidates:
        if set(candidate.get("code") or {}) != fields:
            raise SealError(
                f"candidate {candidate['rank']}: its code needs exactly the fields {sorted(fields)}"
            )
    return {
        "manifest_version": MANIFEST_VERSION,
        "name": name,
        "sealed_at": (sealed_at or datetime.now(UTC)).isoformat(),
        "scene": scene,
        "rules_version": rules_version,
        "schema_version": SCHEMA_VERSION,
        "cutoff": cutoff,
        "strategy_code": strategy_code,
        "threshold": threshold,
        "candidates": candidates,
        "candidates_sha256": candidates_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def seal(
    candidates_path: str | Path,
    *,
    scene: Scene,
    threshold: float,
    policy: CutoffPolicy,
    prereg_dir: str | Path,
    name: str | None = None,
    repo: str | Path | None = None,
    tag: bool = True,
    sign: bool = True,
) -> Sealed:
    """Seal a ranked candidate list for the scene. Manifests are never overwritten."""
    if scene.name != policy.scene:
        raise SealError(f"the cutoff policy is for {policy.scene!r}, not {scene.name!r}")
    if scene.strategy_code is None:
        raise SealError(f"scene {scene.name!r} has no strategy code definition")
    sealed_at = datetime.now(UTC)
    name = name or f"{scene.name}-{sealed_at:%Y%m%dT%H%M%SZ}"
    manifest_path = Path(prereg_dir) / f"{name}{MANIFEST_SUFFIX}"
    if manifest_path.exists() or hash_path(manifest_path).exists():
        raise SealError(f"{manifest_path} already exists; sealed manifests are never replaced")

    candidates_path = Path(candidates_path)
    commit, dirty = _git_state(repo)
    manifest = build_manifest(
        name=name,
        scene=scene.name,
        rules_version=scene.rules_version,
        strategy_code=scene.strategy_code.model_dump(),
        threshold=float(threshold),
        candidates=load_candidates(candidates_path),
        cutoff=policy.cutoff.isoformat(),
        candidates_sha256=sha256_hex(candidates_path.read_bytes()),
        git_commit=commit,
        git_dirty=dirty,
        sealed_at=sealed_at,
    )
    data = manifest_bytes(manifest)
    digest = sha256_hex(data)

    tag_name = make_tag(repo or ".", name, digest, policy, sign=sign) if tag else None
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(data)
    hash_path(manifest_path).write_text(f"{digest}  {manifest_path.name}\n", encoding="utf-8")
    for path in (manifest_path, hash_path(manifest_path)):
        path.chmod(0o444)
    return Sealed(manifest=manifest_path, sha256=digest, tag=tag_name)


def make_tag(repo: str | Path, name: str, digest: str, policy: CutoffPolicy, sign: bool) -> str:
    """Make the git tag prereg/<name>, whose message carries the manifest's SHA-256."""
    tag_name = f"{TAG_PREFIX}{name}"
    message = f"SPAR pre-registration {name}\n\nmanifest sha256 {digest}\ncutoff {policy.cutoff}\n"
    result = run_git(repo, "tag", "-s" if sign else "-a", tag_name, "-m", message)
    if result.returncode != 0:
        hint = " Set up git signing, or pass --no-sign." if sign else ""
        raise SealError(f"git tag {tag_name} failed: {result.stderr.strip()}.{hint}")
    return tag_name


def _git_state(repo: str | Path | None) -> tuple[str | None, bool | None]:
    head = run_git(repo or ".", "rev-parse", "HEAD")
    if head.returncode != 0:
        return None, None
    status = run_git(repo or ".", "status", "--porcelain")
    return head.stdout.strip(), bool(status.stdout.strip())


def run_git(repo: str | Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    except FileNotFoundError:
        return subprocess.CompletedProcess(args, 127, "", "git is not installed")


def _has_todo(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("TODO")
    if isinstance(value, dict):
        return any(_has_todo(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_todo(v) for v in value)
    return False
