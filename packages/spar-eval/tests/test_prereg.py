"""Sealing and verifying a pre-registration, end to end on a dummy candidate list."""

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from spar_eval.cli import app
from spar_eval.prereg.seal import SealError, seal
from spar_eval.prereg.verify import verify
from spar_schema import CutoffPolicy, load_scene, require_processable
from spar_schema.prereg import ManifestError

TOY_SCENE = Path(__file__).parents[2] / "spar-schema" / "tests" / "fixtures" / "toy_scene"
POLICY = CutoffPolicy(scene="toy", cutoff=date(2026, 11, 30))
CANDIDATES = [
    {"rank": 1, "code": {"first_target": ["G2", "G1"]}, "win_rate": 0.71},
    {"rank": 2, "code": {"first_target": ["G1"]}, "win_rate": 0.64},
]
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture
def scene():
    return load_scene(TOY_SCENE, strict=True)


@pytest.fixture
def candidates(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(CANDIDATES), encoding="utf-8")
    return path


def git_repo(path):
    def git(*args):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)

    path.mkdir()
    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "SPAR test")
    git("config", "commit.gpgsign", "false")
    git("config", "tag.gpgsign", "false")
    (path / "README").write_text("test repo\n", encoding="utf-8")
    git("add", "README")
    git("commit", "-q", "-m", "init")
    return path


def test_seal_and_verify_end_to_end(tmp_path, scene, candidates):
    prereg = tmp_path / "prereg"
    sealed = seal(
        candidates, scene=scene, threshold=0.35, policy=POLICY, prereg_dir=prereg,
        name="toy-test", tag=False,
    )  # fmt: skip
    manifest = json.loads(sealed.manifest.read_text(encoding="utf-8"))
    assert manifest["threshold"] == 0.35
    assert manifest["cutoff"] == "2026-11-30"
    assert manifest["candidates"] == CANDIDATES
    assert manifest["strategy_code"]["fields"]["first_target"]["distance"] == "edit"
    result = verify(sealed.manifest)
    assert (result.sha256, result.tag) == (sealed.sha256, "not checked")
    require_processable("2027-01-05", POLICY, prereg_dir=prereg)  # later footage may now be used


def test_tampering_is_caught(tmp_path, scene, candidates):
    sealed = seal(
        candidates, scene=scene, threshold=0.35, policy=POLICY, prereg_dir=tmp_path, tag=False
    )
    sealed.manifest.chmod(0o644)
    text = sealed.manifest.read_text(encoding="utf-8")
    sealed.manifest.write_text(
        text.replace('"threshold": 0.35', '"threshold": 0.5'), encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="changed after sealing"):
        verify(sealed.manifest)


def test_manifests_are_never_replaced(tmp_path, scene, candidates):
    kwargs = dict(
        scene=scene, threshold=0.35, policy=POLICY, prereg_dir=tmp_path, name="once", tag=False
    )
    seal(candidates, **kwargs)
    with pytest.raises(SealError, match="already exists"):
        seal(candidates, **kwargs)


@pytest.mark.parametrize(
    "rows, message",
    [
        ([], "empty"),
        ([{"rank": 2, "code": {"first_target": []}}], "ranked"),
        ([{"rank": 1, "code": {"other_field": []}}], "fields"),
    ],
    ids=["empty list", "ranks out of order", "wrong code fields"],
)
def test_bad_candidate_lists_are_refused(tmp_path, scene, rows, message):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(SealError, match=message):
        seal(path, scene=scene, threshold=0.3, policy=POLICY, prereg_dir=tmp_path, tag=False)
    assert not list(tmp_path.glob("*.manifest.json"))


def test_strategy_code_with_todo_values_is_refused(tmp_path, candidates):
    scene_dir = shutil.copytree(TOY_SCENE, tmp_path / "toy_scene")
    code = scene_dir / "strategy_code.yaml"
    code.write_text(
        code.read_text(encoding="utf-8").replace("weight: 1.0", "weight: TODO"), encoding="utf-8"
    )
    with pytest.raises(SealError, match="TODO"):
        seal(
            candidates,
            scene=load_scene(scene_dir),
            threshold=0.3,
            policy=POLICY,
            prereg_dir=tmp_path,
            tag=False,
        )


def test_policy_for_another_scene_is_refused(tmp_path, scene, candidates):
    other = CutoffPolicy(scene="other", cutoff=date(2026, 11, 30))
    with pytest.raises(SealError, match="other"):
        seal(candidates, scene=scene, threshold=0.3, policy=other, prereg_dir=tmp_path, tag=False)


def test_parquet_candidates_store_codes_as_json(tmp_path, scene):
    path = tmp_path / "candidates.parquet"
    rows = [{"rank": c["rank"], "code": json.dumps(c["code"])} for c in CANDIDATES]
    pq.write_table(pa.Table.from_pylist(rows), path)
    sealed = seal(path, scene=scene, threshold=0.3, policy=POLICY, prereg_dir=tmp_path, tag=False)
    manifest = json.loads(sealed.manifest.read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["code"] == CANDIDATES[0]["code"]


@needs_git
def test_seal_tags_the_hash_in_git(tmp_path, scene, candidates):
    repo = git_repo(tmp_path / "repo")
    sealed = seal(
        candidates, scene=scene, threshold=0.35, policy=POLICY, prereg_dir=tmp_path / "prereg",
        name="toy-tagged", repo=repo, sign=False,
    )  # fmt: skip
    assert sealed.tag == "prereg/toy-tagged"
    result = verify(sealed.manifest, repo=repo)
    assert (result.tag, result.signature, result.ok) == ("ok", "unsigned", True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    manifest = json.loads(sealed.manifest.read_text(encoding="utf-8"))
    assert manifest["git_commit"] == head.stdout.strip()
    assert manifest["git_dirty"] is False


@needs_git
def test_failed_tag_writes_nothing(tmp_path, scene, candidates):
    repo = git_repo(tmp_path / "repo")
    kwargs = dict(scene=scene, threshold=0.35, policy=POLICY, name="dup", repo=repo, sign=False)
    seal(candidates, prereg_dir=tmp_path / "first", **kwargs)
    with pytest.raises(SealError, match="git tag"):
        seal(candidates, prereg_dir=tmp_path / "second", **kwargs)  # the tag already exists
    assert not (tmp_path / "second").exists()


def test_cli_cutoff_seal_and_verify(tmp_path, monkeypatch, candidates):
    shutil.copytree(TOY_SCENE, tmp_path / "scenes" / "toy")
    (tmp_path / "configs" / "eval").mkdir(parents=True)
    (tmp_path / "configs" / "eval" / "prereg.yaml").write_text(
        "scene: toy\ncutoff: 2026-11-30\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPAR_DATA", str(tmp_path / "data"))
    runner = CliRunner()

    before = runner.invoke(app, ["cutoff"])
    assert before.exit_code == 0 and "Not sealed" in before.output

    sealed = runner.invoke(
        app, ["seal", str(candidates), "--threshold", "0.3", "--no-tag", "--name", "cli"]
    )
    assert sealed.exit_code == 0, sealed.output
    assert "Sealed" in sealed.output and "No git tag" in sealed.output

    after = runner.invoke(app, ["cutoff"])
    assert "Sealed: cli.manifest.json" in after.output

    manifest = tmp_path / "data" / "prereg" / "cli.manifest.json"
    checked = runner.invoke(app, ["verify", str(manifest)])
    assert "Hash matches" in checked.output
    assert checked.exit_code == 1  # no git tag, so nothing outside this machine records the seal
