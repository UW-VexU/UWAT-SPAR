"""Pre-registration manifests and the training data cutoff guard."""

from datetime import date, datetime

import pytest

from spar_schema import CutoffError, CutoffPolicy, require_processable
from spar_schema.prereg import (
    ManifestError,
    check_manifest,
    hash_path,
    manifest_bytes,
    sealed_manifest,
    sha256_hex,
)

POLICY = CutoffPolicy(scene="toy", cutoff=date(2026, 11, 30))


def write_manifest(folder, name, content):
    data = manifest_bytes(content)
    path = folder / f"{name}.manifest.json"
    path.write_bytes(data)
    hash_path(path).write_text(f"{sha256_hex(data)}  {path.name}\n", encoding="utf-8")
    return path


def test_manifest_bytes_do_not_depend_on_key_order():
    assert manifest_bytes({"a": 1, "b": [2, 3]}) == manifest_bytes({"b": [2, 3], "a": 1})


def test_check_manifest_returns_its_hash(tmp_path):
    path = write_manifest(tmp_path, "m", {"scene": "toy"})
    assert check_manifest(path) == sha256_hex(path.read_bytes())


def test_edited_manifest_fails_its_check(tmp_path):
    path = write_manifest(tmp_path, "m", {"scene": "toy"})
    path.write_bytes(path.read_bytes().replace(b"toy", b"tay"))
    with pytest.raises(ManifestError, match="changed after sealing"):
        check_manifest(path)


def test_manifest_without_a_hash_file_was_never_sealed(tmp_path):
    path = tmp_path / "m.manifest.json"
    path.write_bytes(manifest_bytes({"scene": "toy"}))
    with pytest.raises(ManifestError, match="never sealed"):
        check_manifest(path)


@pytest.mark.parametrize(
    "day", ["2026-10-01", "2026-11-30", date(2026, 11, 30), datetime(2026, 11, 30, 18, 0)]
)
def test_footage_up_to_the_cutoff_day_is_allowed(tmp_path, day):
    require_processable(day, POLICY, prereg_dir=tmp_path)


def test_later_footage_is_blocked_until_sealed(tmp_path):
    with pytest.raises(CutoffError, match="2026-12-01"):
        require_processable("2026-12-01", POLICY, prereg_dir=tmp_path)
    write_manifest(tmp_path, "toy-1", {"scene": "toy"})
    require_processable("2026-12-01", POLICY, prereg_dir=tmp_path)
    assert sealed_manifest(tmp_path, "toy").name == "toy-1.manifest.json"


def test_manifest_for_another_scene_does_not_count(tmp_path):
    write_manifest(tmp_path, "other-1", {"scene": "other"})
    with pytest.raises(CutoffError):
        require_processable("2027-01-05", POLICY, prereg_dir=tmp_path)


def test_edited_manifest_does_not_count(tmp_path):
    path = write_manifest(tmp_path, "toy-1", {"scene": "toy", "threshold": 0.3})
    path.write_bytes(path.read_bytes().replace(b"0.3", b"0.4"))
    with pytest.raises(CutoffError):
        require_processable("2027-01-05", POLICY, prereg_dir=tmp_path)


def test_default_folder_is_under_spar_data(tmp_path, monkeypatch):
    monkeypatch.setenv("SPAR_DATA", str(tmp_path))
    with pytest.raises(CutoffError, match="prereg"):
        require_processable("2027-01-05", POLICY)
    monkeypatch.delenv("SPAR_DATA")
    with pytest.raises(RuntimeError, match="SPAR_DATA is not set"):
        require_processable("2027-01-05", POLICY)


def test_policy_loads_from_yaml(tmp_path):
    (tmp_path / "prereg.yaml").write_text("scene: toy\ncutoff: 2026-11-30\n", encoding="utf-8")
    assert CutoffPolicy.load(tmp_path / "prereg.yaml") == POLICY
