"""The plan-level action set a scene declares in its `sim` section (spec section 6.7).

spar-state labels episodes with these actions (actions.parquet), and spar-sim runs them. Each
action has named arguments, and each argument takes one label from a fixed list. action_set()
resolves the scene's lists for one program, and gives the labels of a destination argument their
field positions, so a simulator knows where the robot drives:

    actions = action_set(scene, "solo")
    code, args = actions.encode("place", {"goal": "red"})   # 2, [0]
    actions.decode(code, args)                              # ("place", {"goal": "red"})
"""

from dataclasses import dataclass
from typing import Any

from spar_schema.scene import Scene, SceneError

Point = tuple[float, float]


@dataclass(frozen=True)
class ActionArg:
    name: str
    labels: tuple[str, ...]
    positions: tuple[Point, ...] | None = None  # per label, when every label is a place
    opponents: bool = False  # labels number the other alliance's robots, which move


@dataclass(frozen=True)
class Action:
    name: str
    args: tuple[ActionArg, ...]
    to_arg: int | None = None  # the argument giving the destination
    to_xy: Point | None = None  # or a fixed destination
    standoff: bool = False  # stop short of the destination (object instances and robots)

    @property
    def moves(self) -> bool:
        return self.to_arg is not None or self.to_xy is not None


@dataclass(frozen=True)
class ActionSet:
    actions: tuple[Action, ...]
    items: tuple[str, ...]
    carry: int
    standoff_m: float

    @property
    def names(self) -> list[str]:
        return [action.name for action in self.actions]

    @property
    def max_args(self) -> int:
        return max((len(action.args) for action in self.actions), default=0)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no action {name!r}; the scene has {self.names}") from None

    def __getitem__(self, name: str) -> Action:
        return self.actions[self.index(name)]

    def encode(self, name: str, args: dict[str, Any] | None = None) -> tuple[int, list[int]]:
        """An action's index and its argument codes, padded with 0 to max_args.

        Keys of args that are not arguments of the action (notes such as an object ID) are ignored.
        """
        action = self[name]
        args = args or {}
        codes = []
        for arg in action.args:
            if arg.name not in args:
                raise KeyError(f"{name} needs argument {arg.name!r}")
            label = str(args[arg.name])
            if label not in arg.labels:
                raise KeyError(f"{name}: {arg.name} is one of {list(arg.labels)}, not {label!r}")
            codes.append(arg.labels.index(label))
        return self.index(name), codes + [0] * (self.max_args - len(codes))

    def decode(self, code: int, args: list[int] | tuple[int, ...]) -> tuple[str, dict[str, str]]:
        action = self.actions[int(code)]
        return action.name, {
            arg.name: arg.labels[int(value)] for arg, value in zip(action.args, args, strict=False)
        }


def action_set(scene: Scene, program: str) -> ActionSet:
    """The scene's action set for one program, with every label list resolved."""
    sim = scene.spec.sim
    if sim is None:
        raise SceneError(f"scene {scene.name!r} has no sim section")
    if program not in scene.programs:
        raise SceneError(f"scene {scene.name!r} has no program {program!r}")
    regions = scene.regions
    instances = scene.instance_positions
    places = {name: _centroid(polygon) for name, polygon in regions.items()} | instances
    opponents = tuple(str(i + 1) for i in range(scene.programs[program].robots_per_alliance))

    def resolve(action: str, name: str, domain: list[str] | str) -> ActionArg:
        if isinstance(domain, list):
            labels = tuple(str(label) for label in domain)
        elif domain == "items":
            labels = tuple(sim.items)
        elif domain == "opponents":
            return ActionArg(name, opponents, opponents=True)
        elif "instances" in scene.spec.objects.get(domain, {}):
            labels = tuple(scene.spec.objects[domain]["instances"])
        else:
            raise SceneError(
                f"{scene.name}: action {action}, argument {name}: {domain!r} is not a list, "
                "'items', 'opponents' or an object class with instances"
            )
        if not labels:
            raise SceneError(f"{scene.name}: action {action}, argument {name} has no labels")
        positions = tuple(places[label] for label in labels) if set(labels) <= set(places) else None
        return ActionArg(name, labels, positions)

    actions = []
    for name, spec in sim.actions.items():
        args = tuple(resolve(name, arg, domain) for arg, domain in spec.args.items())
        to_arg = to_xy = None
        standoff = False
        if spec.to is not None:
            arg_names = [arg.name for arg in args]
            if spec.to in arg_names:
                to_arg = arg_names.index(spec.to)
                arg = args[to_arg]
                if arg.positions is None and not arg.opponents:
                    raise SceneError(f"{scene.name}: action {name} goes to {spec.to}, not places")
                standoff = arg.opponents or set(arg.labels) <= set(instances)
            elif spec.to in places:
                to_xy = places[spec.to]
                standoff = spec.to in instances
            else:
                raise SceneError(
                    f"{scene.name}: action {name} goes to {spec.to!r}, which is not one of its "
                    "arguments, a field region or an object instance"
                )
        actions.append(Action(name, args, to_arg, to_xy, standoff))
    return ActionSet(tuple(actions), tuple(sim.items), sim.carry, sim.standoff_m)


def _centroid(polygon: list) -> Point:
    """The mean of a polygon's corners: inside it for the convex regions scenes use."""
    return (
        sum(float(p[0]) for p in polygon) / len(polygon),
        sum(float(p[1]) for p in polygon) / len(polygon),
    )
