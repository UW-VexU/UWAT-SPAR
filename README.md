# SPAR: Synthetic Play from Archived Replays

SPAR turns livestream video of VEX robot matches into multi-robot training data, checks that data
against the official match scores, and combines it with simulation to train robot policies and to
search for strategies nobody has played yet. The test case is VEX Override (2026 to 2027 season).

- Full technical design: [docs/SPAR_Repo_Spec_V1.md](docs/SPAR_Repo_Spec_V1.md)
- Research proposal: [docs/SPAR.pdf](docs/SPAR.pdf)

**Status:** skeleton. Every package and module exists as a stub, and the workspace installs,
lints and tests cleanly. The only working code is the original auto-labeler, waiting to be
ported (see [Legacy auto-labeler](#legacy-auto-labeler)).

## Quick start

You need [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/). uv downloads the right
Python (3.12) by itself.

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh                          # Linux or macOS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# Get the code and install everything
git clone https://github.com/UW-VexU/UWAT-SPAR.git
cd UWAT-SPAR
uv sync

# Check it works
uv run spar --help
uv run pytest
```

`uv sync` creates `.venv/` and installs every workspace package in editable mode, so code changes
take effect without reinstalling. Run it again after pulling or after changing any
`pyproject.toml`.

## Everyday commands

| Command | What it does |
|---|---|
| `uv sync` | Install or update `.venv` to match `uv.lock` |
| `uv run spar --help` | List the command groups. `uv run spar ingest --help` lists one group's commands |
| `uv run pytest` | Run every test (`tests/` and each `packages/*/tests/`) |
| `uv run pytest packages/spar-schema` | Run one package's tests |
| `uv run pytest tests/test_workspace.py -k cli` | Run one file, filtered by test name |
| `uv run ruff check` | Lint. Add `--fix` to apply safe fixes |
| `uv run ruff format` | Format the code |
| `uv add --package spar-ingest yt-dlp` | Add a dependency to one package (updates its `pyproject.toml` and `uv.lock`) |

Before you push, run these checks. CI is not set up yet, but it will run the same three:

```bash
uv run ruff check && uv run ruff format --check && uv run pytest
```

## Testing

| Where | What goes there |
|---|---|
| `packages/spar-<name>/tests/` | Unit tests for that package. Use tiny synthetic tables built in the test, not real data |
| `tests/` | Cross-package and end-to-end tests |
| `tests/fixtures/` | Small files checked into git. `tests/fixtures/frames/` holds 8 livestream screenshots |

- Tests must run on a CPU, without `$SPAR_DATA` and without network access, because CI runs on
  CPU-only GitHub runners.
- Larger test data, such as the 30 s sample match with hand labels, is fetched by
  `scripts/fetch_test_data.py` (not written yet) and never committed.
- pytest uses `--import-mode=importlib`, so test files in different packages may share a name.
- Right now the only test is `tests/test_workspace.py`. It checks that every package imports and
  that `spar` mounts every command group. A package with no tests prints `no tests ran`.

## How the repo is organized

```text
UWAT-SPAR/
├── pyproject.toml     workspace root: member list, dev tools, ruff and pytest settings
├── uv.lock            exact versions of every dependency, committed
├── scenes/            game rules, one folder per game (override, pushback, pinnacle)
├── packages/          the Python packages, one per pipeline stage (table below)
├── render/            Blender rendering, its own environment (not in the uv workspace)
├── configs/           Hydra configs: perception/, learn/, search/, eval/
├── docker/            main.Dockerfile and render.Dockerfile (placeholders for now)
├── scripts/           one-off utilities, such as fetching test data
├── tests/             cross-package tests and small fixtures
└── docs/              spec, proposal, scene authoring guide, data format guide
```

### The packages

The pipeline runs top to bottom. Each stage reads the previous stage's files and writes its own.

| Package | Command | Job | Spec |
|---|---|---|---|
| `spar-schema` | none | Shared contract: scene loading, episode tables, reader and writer. Used by every package and by the physics repo | 5, 6.1 |
| `spar-ingest` | `spar ingest` | Match records from the VEX Events API, stream download, cutting streams into matches | 6.2 |
| `spar-perception` | `spar perception` | Auto-labeling, detector training, tracking, field calibration | 6.3 |
| `spar-state` | `spar state` | Task state, score check against the official score, events, strategy codes | 6.4 |
| `spar-data` | `spar data` | Episode catalog, start states, export to LeRobot, PettingZoo and others | 6.5 |
| `spar-gt` | `spar gt` | Ground truth from our own scrimmages (AprilTags, robot logs) | 6.6 |
| `spar-sim` | `spar sim` | Fast plan-level simulator in JAX | 6.7 |
| `spar-learn` | `spar learn` | Behavior cloning, multi-agent RL, multi-run strategy search | 6.8 |
| `spar-eval` | `spar eval` | Metrics, statistics, sealing the candidate list | 6.9 |
| `spar-cli` | `spar` | Mounts every package's commands under one `spar` command | 4 |

### Inside a package

Every package has the same layout. The folder name uses dashes and the import name uses
underscores: `packages/spar-ingest` is imported as `spar_ingest`.

```text
packages/spar-<name>/
├── pyproject.toml          name, dependencies, workspace sources
├── src/spar_<name>/
│   ├── __init__.py         public API only
│   ├── cli.py              Typer app, mounted as `spar <name>`
│   └── <module>.py         one module per job; each stub's docstring says what it is for
└── tests/
```

### Rules that keep it general

1. **Nothing outside `scenes/` may mention Override or any other game.** Classes, prompts, field
   geometry, scoring rules and strategy code fields are data or plugins loaded from a scene
   folder. Adding a game means adding a scene folder, not editing package code.
2. **Stages talk only through files in the formats from `spar-schema`.** Packages depend on
   `spar-schema` and never import each other's internals. The one exception is `spar-learn`,
   which runs `spar-sim` in-process. This keeps every stage testable on its own.
3. **`spar-schema` stays light.** It depends only on Pydantic, PyArrow and PyYAML, and supports
   Python 3.11 and up, so the physics repo and the render environment can install it.
4. **AGPL and GPL code stays out of the core path.** Ultralytics and BoxMOT go in an optional
   `baselines` extra. `bpy` is used only in `render/`.
5. **Training data cutoff.** Footage recorded after November 30, 2026 may be downloaded but must
   not be processed by any `spar` command until `spar eval seal` has run.

### Scenes

Each game is a folder in `scenes/`. `scenes/override/` is the reference:

| File | Contents |
|---|---|
| `scene.yaml` | Field, objects, programs, timing, teacher prompts, objective |
| `scoring.py` | `score(state, xp)`: points per alliance, written with array operations so it runs in NumPy and JAX |
| `strategy_code.yaml` | Strategy code fields, distances, weights, coder threshold |
| `landmarks.yaml` | Named field points for calibration |
| `robot_limits.yaml` | Our robots' speed and carry limits, used only by the search |
| `README.md` | Plain-language rules summary with manual links |

Values marked `TODO` still have to be filled in from the game manual.

### render/

`render/` imports Blender as the `bpy` module, and each bpy release pins one Python version, so it
has its own `pyproject.toml` and environment and runs from `docker/render.Dockerfile`. It shares
only files with the main environment. See [render/README.md](render/README.md).

## Data

All data lives under one folder outside git, set by the `SPAR_DATA` environment variable:

```bash
export SPAR_DATA=~/spar-data            # Linux or macOS
$env:SPAR_DATA = "D:\spar-data"         # Windows PowerShell
```

```text
$SPAR_DATA/
├── registry/     events, matches and streams tables
├── raw/streams/  downloaded livestreams
├── frames/       sampled frames for labeling
├── labels/       teacher/, hand/, render/ (COCO segmentation)
├── models/       detector/, landmarks/, event_model/
├── episodes/     <scene>/<rules_version>/<program>/<event_sku>/<episode_id>/
├── renders/      rendered matches
├── gt/           scrimmage ground truth
├── runs/         Hydra outputs, search archives, checkpoints
└── prereg/       sealed manifests and hashes
```

Never commit video, datasets or model weights. `.gitignore` already blocks `*.mp4`, `*.pt`,
`*.onnx` and `*.safetensors`. Keep API tokens in a `.env` file, which is also ignored. The
episode format is described in section 5.5 of the spec.

## Adding code

**A command.** Add a function to the package's `cli.py`:

```python
@app.command()
def events(season: str):
    """Pull the event list for a season."""
    # Import heavy modules inside the function so `spar --help` stays fast.
    from spar_ingest.events_api import fetch_events

    fetch_events(season)
```

It then runs as `uv run spar ingest events --season override`.

**A dependency.** Add it to the package that uses it, not the workspace root:

```bash
uv add --package spar-perception transformers
uv add --package spar-perception --optional baselines ultralytics   # AGPL, optional extra only
```

Section 10 of the spec lists the planned libraries and versions. Each package's `pyproject.toml`
also has a `# Planned` comment listing its own.

**A game.** Copy the file set from `scenes/override/` into `scenes/<game>/` and fill it in. See
[docs/scene_authoring.md](docs/scene_authoring.md).

## Command line

The planned commands, from the spec. The groups exist and are empty:

```bash
spar ingest      events, matches, streams, download, cut, count
spar perception  autolabel, review, train-detector, run
spar state       run
spar data        catalog, start-states, convert, publish
spar gt          build
spar sim         fit
spar learn       bc, search, exploit, export
spar eval        seal, verify, novelty, report
```

Section 7 of the spec shows a full run from match records to the sealed candidate list.

## Legacy auto-labeler

The original scripts were moved unchanged into `spar-perception`. They will become
`spar perception autolabel` and `spar perception review` (spec section 11).

| File | Was | Does |
|---|---|---|
| `autolabel/pipeline.py` | `main.py` | Finds robots with Grounding DINO (prompt "A Vex Robot.", box threshold 0.23, text threshold 0.35), outlines them with SAM 2, writes one YOLO segmentation label file per image. Set `BOXES_MODE = True` to write boxes instead of masks |
| `autolabel/review.py` | `checker.py` | Draws the labeled outlines on each image, with the background blacked out, so a person can check them |

Until they are ported:

- They read `frames/*.jpg` and write `labels/` and `checked/` **relative to the current folder**,
  so run them from a working folder that contains a `frames/` folder, not from the repo root.
- They need torch, transformers, ultralytics, pillow and opencv-python, which the workspace does
  not install yet. One way to run them is with uv adding those for the one run. This gets the CPU
  build of PyTorch. For the GPU, install a CUDA 12.8+ build instead.

  ```bash
  cd path/to/workdir      # contains frames/*.jpg
  uv run --no-project --with torch --with transformers --with ultralytics --with pillow \
      python path/to/UWAT-SPAR/packages/spar-perception/src/spar_perception/autolabel/pipeline.py
  uv run --no-project --with opencv-python --with numpy \
      python path/to/UWAT-SPAR/packages/spar-perception/src/spar_perception/autolabel/review.py
  ```

- The first run downloads the Grounding DINO weights from Hugging Face and `sam2_b.pt` into the
  current folder.

## Timeline

| Milestone | Date |
|---|---|
| Foundations: workspace, schema, Override scene spec, ingest | Nov 8, 2026 |
| Perception works on real matches | Nov 29, 2026 |
| Training data cutoff | Nov 30, 2026 |
| Candidate list sealed | Jan 10, 2027 |
| Paper submitted (Autonomous Robots, Synthetic Data for Robot Learning) | Feb 1, 2027 |

The full build plan and the open items are in sections 12 and 13 of the spec.
