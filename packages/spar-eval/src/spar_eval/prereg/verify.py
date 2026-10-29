"""Checks a manifest against its sealed hash.

verify() checks the manifest against its .sha256 file and, given the repo, that the git tag
prereg/<name> exists, carries the same hash and has a valid signature. It also reports whether an
OpenTimestamps proof (<manifest>.ots) sits beside the manifest.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from spar_eval.prereg.seal import TAG_PREFIX, run_git
from spar_schema.prereg import check_manifest


@dataclass(frozen=True)
class Verification:
    manifest: Path
    sha256: str  # the hash both files agree on (verify raises ManifestError if they do not)
    tag: str  # "ok", "missing", "wrong hash" or "not checked"
    signature: str  # "valid", "invalid", "unsigned" or "not checked"
    timestamp_proof: bool  # an OpenTimestamps .ots file is beside the manifest

    @property
    def ok(self) -> bool:
        return self.tag in ("ok", "not checked") and self.signature != "invalid"


def verify(manifest_path: str | Path, repo: str | Path | None = None) -> Verification:
    path = Path(manifest_path)
    digest = check_manifest(path)
    name = json.loads(path.read_text(encoding="utf-8"))["name"]
    tag, signature = "not checked", "not checked"
    if repo is not None:
        tag_name = f"{TAG_PREFIX}{name}"
        if run_git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag_name}").returncode != 0:
            tag = "missing"
        else:
            message = run_git(repo, "tag", "-l", "--format=%(contents)", tag_name).stdout
            tag = "ok" if f"manifest sha256 {digest}" in message else "wrong hash"
            raw = run_git(repo, "cat-file", "tag", tag_name).stdout
            if "-----BEGIN" not in raw:
                signature = "unsigned"
            else:
                signed = run_git(repo, "tag", "-v", tag_name).returncode == 0
                signature = "valid" if signed else "invalid"
    return Verification(
        manifest=path,
        sha256=digest,
        tag=tag,
        signature=signature,
        timestamp_proof=Path(str(path) + ".ots").is_file(),
    )
