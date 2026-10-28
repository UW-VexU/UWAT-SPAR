"""SceneSpec Pydantic model, load_scene(path, rules_version=None) and the scoring plugin loader.

A scene folder holds everything specific to one game (spec section 5.2):

    scenes/<game>/
    ├── scene.yaml          SceneSpec: field, programs, timing, agents, objects, objective,
    │                       prompts, and the TaskStateSpec that score() reads
    ├── scoring.py          score(state, xp) -> points per alliance
    ├── strategy_code.yaml  StrategyCodeSpec
    ├── landmarks.yaml      LandmarksSpec: named points in the field frame (m)
    └── robot_limits.yaml   our robots' limits

While a scene is being written, any value may be the placeholder "TODO" (or a string starting
with it). load_scene() lists those values in Scene.todos. Code that writes data must not use an
incomplete scene: load it with strict=True, or call Scene.require_complete().
"""

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from spar_schema.task_state import TaskState, TaskStateSpec
from spar_schema.version import check_compatible

SCENE_FILE = "scene.yaml"
TODO_PREFIX = "TODO"

# A value that has not been filled in yet.
Todo = Annotated[str, StringConstraints(pattern=r"^TODO")]

M = TypeVar("M", bound=BaseModel)


class SceneError(ValueError):
    """A scene folder is missing a file or does not match SceneSpec."""


class SceneIncompleteError(SceneError):
    """A scene still has TODO values where complete data is required."""


class SceneVersionError(SceneError):
    """A scene has a different rules version from the one requested."""


class _Spec(BaseModel):
    # Unknown keys are errors, so a typo in a scene file is caught when it loads.
    model_config = ConfigDict(extra="forbid")


class FrameSpec(_Spec):
    origin: str
    x_axis: str


class FieldSpec(_Spec):
    size_m: tuple[float, float]
    frame: FrameSpec
    landmarks: str  # file in the scene folder
    regions: dict[str, Any] = {}  # polygons in the field frame


class ProgramSpec(_Spec):
    teams_per_alliance: int
    robots_per_alliance: int
    autonomous_s: float | Todo
    driver_s: float | Todo
    robot_sizes_in: list[float] | None = None


class AgentsSpec(_Spec):
    classes: list[str] = Field(min_length=1)
    alliances: list[str] = Field(min_length=1)
    our_robots: str | None = None  # robot limits file in the scene folder


class ObjectiveSpec(_Spec):
    module: str  # scoring plugin in the scene folder
    function: str = "score"
    checkpoints: list[str] = Field(min_length=1)
    points: dict[str, float | Todo] = {}


class SceneSpec(_Spec):
    """scene.yaml."""

    scene: str
    rules_version: str
    schema_version: str
    manual: str | None = None
    field: FieldSpec
    programs: dict[str, ProgramSpec] = Field(min_length=1)
    timing: dict[str, float | Todo] = {}
    agents: AgentsSpec
    objects: dict[str, dict[str, Any]] = {}
    objective: ObjectiveSpec
    prompts: dict[str, str] = {}  # text prompt per class for the teacher models
    strategy_code: str | None = None  # file in the scene folder
    state: TaskStateSpec | None = None  # the arrays score() reads


class StrategyFieldSpec(BaseModel):
    """One strategy code field. Other keys (length, parts, missing_penalty) depend on the type."""

    model_config = ConfigDict(extra="allow")

    type: str
    distance: str
    weight: float | Todo = 1.0


class StrategyCodeSpec(_Spec):
    """strategy_code.yaml."""

    fields: dict[str, StrategyFieldSpec]
    threshold: dict[str, Any] = {}


class LandmarksSpec(_Spec):
    """landmarks.yaml: named points in the field frame (m)."""

    landmarks: dict[str, tuple[float, float, float] | Todo] = {}


@dataclass(frozen=True)
class Scene:
    """A loaded scene: the validated spec, the files it references and its scoring function."""

    root: Path
    spec: SceneSpec
    landmarks: dict[str, tuple[float, float, float] | str]
    robot_limits: dict[str, Any]
    strategy_code: StrategyCodeSpec | None
    score_fn: Callable[..., dict[str, Any]]
    todos: tuple[str, ...]  # "<file>: <dotted.path>" for every value still TODO

    @property
    def name(self) -> str:
        return self.spec.scene

    @property
    def rules_version(self) -> str:
        return self.spec.rules_version

    @property
    def programs(self) -> dict[str, ProgramSpec]:
        return self.spec.programs

    @property
    def alliances(self) -> list[str]:
        return self.spec.agents.alliances

    @property
    def object_classes(self) -> list[str]:
        return list(self.spec.objects)

    @property
    def checkpoints(self) -> list[str]:
        return self.spec.objective.checkpoints

    @property
    def task_state(self) -> TaskStateSpec:
        """The arrays score() reads, from the scene's `state` section."""
        if self.spec.state is None:
            raise SceneError(f"scene {self.name!r} has no state section")
        return self.spec.state

    def score(self, state: TaskState, xp: Any) -> dict[str, Any]:
        """Points per alliance for one state, from the scene's scoring plugin.

        xp is the array module (numpy or jax.numpy). Values are 0-d arrays, so the call also works
        under jax.jit; use float() on them outside JAX.
        """
        return self.score_fn(state, xp)

    def require_complete(self) -> None:
        """Raise SceneIncompleteError if any value is still TODO."""
        if self.todos:
            raise SceneIncompleteError(
                f"scene {self.name!r} still has TODO values: {', '.join(self.todos)}"
            )


def load_scene(
    path: str | Path, rules_version: str | None = None, *, strict: bool = False
) -> Scene:
    """Load and validate a scene folder, or the scene.yaml inside it.

    rules_version, when given, must match the scene's, so data is never read under the wrong
    rules. strict=True also refuses a scene that still has TODO values.
    """
    root = Path(path)
    if root.is_file():
        root = root.parent
    raw = _read_yaml(root / SCENE_FILE)
    check_compatible(str(raw.get("schema_version", "")))
    spec = _validate(SceneSpec, raw, root / SCENE_FILE)
    if rules_version is not None and rules_version != spec.rules_version:
        raise SceneVersionError(
            f"{root}: scene has rules version {spec.rules_version!r}, not {rules_version!r}"
        )
    todos = _find_todos(raw, SCENE_FILE)

    raw_landmarks = _read_yaml(root / spec.field.landmarks)
    landmarks = _validate(LandmarksSpec, raw_landmarks, root / spec.field.landmarks).landmarks
    todos += _find_todos(raw_landmarks, spec.field.landmarks)

    robot_limits: dict[str, Any] = {}
    if spec.agents.our_robots:
        robot_limits = _read_yaml(root / spec.agents.our_robots)
        todos += _find_todos(robot_limits, spec.agents.our_robots)

    strategy_code = None
    if spec.strategy_code:
        raw_code = _read_yaml(root / spec.strategy_code)
        strategy_code = _validate(StrategyCodeSpec, raw_code, root / spec.strategy_code)
        todos += _find_todos(raw_code, spec.strategy_code)

    scene = Scene(
        root=root,
        spec=spec,
        landmarks=landmarks,
        robot_limits=robot_limits,
        strategy_code=strategy_code,
        score_fn=_load_score(root, spec),
        todos=tuple(todos),
    )
    if strict:
        scene.require_complete()
    return scene


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SceneError(f"{path} not found")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SceneError(f"{path}: expected a mapping at the top level")
    return data


def _validate(model: type[M], data: dict[str, Any], path: Path) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise SceneError(f"{path}: {exc}") from exc


def _find_todos(data: Any, filename: str) -> list[str]:
    """The dotted path of every value in `data` that is still TODO."""
    found = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, str) and value.startswith(TODO_PREFIX):
            found.append(f"{filename}: {path}")
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, f"{path}[{i}]")

    walk(data, "")
    return found


def _load_score(root: Path, spec: SceneSpec) -> Callable[..., dict[str, Any]]:
    """Import the scene's scoring plugin and return its score function."""
    path = root / spec.objective.module
    if not path.is_file():
        raise SceneError(f"{path} not found")
    module_spec = importlib.util.spec_from_file_location(
        f"spar_scene_{spec.scene}_{path.stem}", path
    )
    if module_spec is None or module_spec.loader is None:
        raise SceneError(f"{path}: cannot be imported")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    score = getattr(module, spec.objective.function, None)
    if not callable(score):
        raise SceneError(f"{path} has no function {spec.objective.function!r}")
    return score
