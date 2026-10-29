"""Override simulator plugin: what each plan-level action in scene.yaml does (spar-sim).

infer_action() names the action behind one event, for spar-state's actions.py. rules(scene,
program) gives spar-sim the rest:
    encode(start, agents)               the task arrays at a StartState
    effects[action](task, ctx, xp)      the arrays after an action finishes, and whether it could
                                        be done; actions without an entry change nothing here
    on_checkpoint(task, c, scores, xp)  records the autonomous winner at autonomous_end
    task_state(task, robots, c, xp)     the TaskState scoring.py reads, at checkpoint c

The task arrays are the TaskState's Goal and Toggle fields, plus:
    height[goal]       levels in use on each Goal; the next object goes on top
    supply[item]       objects of each item that are not on a Goal or held
    held[agent, item]  what each robot carries
    autonomous_winner  its code in scene.yaml; winner_known says whether it is decided yet
Effects use array operations only, through xp, so they run under jax.jit and jax.vmap.

What the actions do:
  - place needs the item in hand and a free level, and is refused on the Midfield Goal in the
    endgame (SG12). A Pin goes on with its first named color down ("pin_red_yellow" puts red
    down) unless flipped; a flipped Cup has its opaque half down.
  - remove takes the top object off a Goal and returns it to the supply.
  - flip_toggle sets a Toggle to the robot's alliance (SC4).
  - pick and load take an item from the supply when the robot has room. The supply is one pool,
    so Loaders and the floor are not told apart.
Robots count as in the Midfield as state.py counts them: within half the program's smallest
starting size of it.
"""

import numpy as np

from spar_schema.frames import inches, polygon_distance, polygon_distances
from spar_schema.task_state import Snapshot

EMPTY, PIN, CUP = 0, 1, 2  # kind codes in scene.yaml
COLOR_ORDER = ["red", "blue", "yellow"]  # how item names list a Pin's colors
LOADER_REACH_M = 0.6  # a pick this close to a Loader is a load


def infer_action(event, obj, xy, scene):
    """The (action, args) behind one event, or None. obj is the object's row at the event, with
    attrs parsed, and xy the robot's position."""
    kind = event["type"]
    if kind in ("pick", "place") and obj is not None:
        item = item_of(obj, scene.spec.sim.items)
        if item is None:
            return None
        if kind == "place":
            return "place", {"goal": event["goal_id"], "item": item, "flip": flip_of(obj)}
        loader = _nearest(xy, scene.spec.objects["loader"]["instances"], LOADER_REACH_M)
        return ("load", {"loader": loader, "item": item}) if loader else ("pick", {"item": item})
    if kind == "remove" and event["goal_id"]:
        return "remove", {"goal": event["goal_id"]}
    if kind == "toggle_flip":
        return "flip_toggle", {"toggle": event["target_id"]}
    if kind == "enter_midfield":
        return "park_midfield", {}
    if kind == "leave_midfield" and xy is not None:
        quadrants = scene.spec.field.regions["quadrants"]
        region = min(quadrants, key=lambda name: polygon_distance(*xy, quadrants[name]))
        return "move_to", {"region": region}
    return None


def item_of(obj, items):
    """The sim item for a Pin or Cup row, a Pin by its two colors in COLOR_ORDER, or None if the
    scene has no such item."""
    if obj["cls"] == "cup":
        return "cup"
    halves = (obj.get("attrs") or {}).get("halves")
    if obj["cls"] != "pin" or not halves:
        return None
    name = "pin_" + "_".join(sorted(halves, key=COLOR_ORDER.index))
    return name if name in items else None


def flip_of(obj):
    attrs = obj.get("attrs") or {}
    if obj["cls"] == "cup":
        return "yes" if attrs.get("gray") == "down" else "no"
    halves = attrs["halves"]
    return "no" if halves[0] == sorted(halves, key=COLOR_ORDER.index)[0] else "yes"


def _nearest(xy, instances, reach):
    if xy is None:
        return None
    distance = {
        name: np.hypot(xy[0] - i["xy"][0], xy[1] - i["xy"][1]) for name, i in instances.items()
    }
    best = min(distance, key=distance.get)
    return best if distance[best] <= reach else None


def rules(scene, program):
    return Rules(scene, program)


class Rules:
    def __init__(self, scene, program):
        self.scene, self.program = scene, program
        spec = self.spec = scene.task_state
        self.items = list(scene.spec.sim.items)
        self.goals, self.levels = spec.dims["goal"], spec.size("level")
        self.quadrants = spec.dims["quadrant"]
        colors = spec.fields["pin_color"].values
        self.cup = self.items.index("cup")
        # Each item's halves as named, in pin_color codes, and the item for a pair of codes.
        self.item_halves = np.zeros((len(self.items), 2), np.int8)
        self.pin_item = np.zeros((len(colors), len(colors)), np.int8)
        self.totals = np.zeros(len(self.items), np.int16)
        for i, name in enumerate(self.items):
            if name.startswith("pin_"):
                a, b = (colors.index(c) for c in name[4:].split("_"))
                self.item_halves[i] = a, b
                self.pin_item[a, b] = self.pin_item[b, a] = i
                self.totals[i] = scene.spec.objects["pin"]["types"][name[4:]]
        self.totals[self.cup] = scene.spec.objects["cup"]["count"]
        toggles = scene.spec.objects["toggle"]["instances"]
        self.toggle_quadrant = np.array(
            [self.quadrants.index(t["quadrant"]) for t in toggles.values()]
        )
        self.alliance_face = np.array([colors.index(a) for a in scene.alliances], np.int8)
        self.alliance_slot = [scene.alliances.index(a) for a in spec.dims["alliance"]]
        self.midfield_goal = self.goals.index("M")
        periods = scene.programs[program]
        play_end = float(periods.autonomous_s) + float(periods.driver_s)
        self.endgame_starts = play_end - float(scene.spec.timing["endgame_s"])
        self.reach = inches(min(periods.robot_sizes_in)) / 2
        self.midfield = scene.regions["midfield"]
        self.autonomous_end = scene.checkpoints.index("autonomous_end")
        self.checkpoint_codes = np.array(
            [spec.code("checkpoint", c) for c in scene.checkpoints], np.int8
        )
        self.winner_codes = {
            w: spec.code("autonomous_winner", w) for w in ("none", "red", "blue", "tie")
        }
        self.effects = {
            "pick": self.pick,
            "load": self.load,
            "place": self.place,
            "remove": self.remove,
            "flip_toggle": self.flip_toggle,
        }  # fmt: skip

    def encode(self, start, agents):
        snapshot = Snapshot(
            t=start.t,
            checkpoint="match_end",
            program=self.program,
            autonomous_winner=start.autonomous_winner or "none",
            robots=[],
            objects=start.objects,
        )
        built = self.scene.build_state(snapshot)
        task = {
            k: built[k]
            for k in (
                "kind",
                "pin_color",
                "cup_opaque_lower",
                "crowded",
                "toggle_face",
                "toggle_set",
            )
        }
        used = (built["kind"] != EMPTY) | built["crowded"]
        top = np.where(used, np.arange(self.levels) + 1, 0).max(axis=1)
        held = np.zeros((len(agents), len(self.items)), np.int8)
        taken = np.zeros(len(self.items), np.int16)
        for obj in start.objects:
            item = item_of(obj, self.items) if obj["cls"] in ("pin", "cup") else None
            if item is None or obj["status"] not in ("placed", "held"):
                continue
            taken[self.items.index(item)] += 1
            if obj["status"] == "held" and obj.get("holder") in agents:
                held[agents.index(obj["holder"]), self.items.index(item)] += 1
        winner = start.autonomous_winner
        return task | {
            "height": top.astype(np.int16),
            "supply": np.maximum(self.totals - taken, 0).astype(np.int16),
            "held": held,
            "autonomous_winner": np.asarray(self.winner_codes[winner or "none"], np.int8),
            "winner_known": np.asarray(winner is not None),
        }

    # Effects. ctx has agent, alliance, args, t, carry and xy (every robot's position).

    def place(self, task, ctx, xp):
        goal, item, flip = ctx["args"][0], ctx["args"][1], ctx["args"][2] == 1
        height = task["height"][goal]
        ok = (task["held"][ctx["agent"], item] > 0) & (height < self.levels)
        ok = ok & ~((goal == self.midfield_goal) & (ctx["t"] > self.endgame_starts))
        on_goal = xp.arange(len(self.goals)) == goal
        at = on_goal[:, None] & (xp.arange(self.levels) == height)[None, :] & ok
        is_cup = item == self.cup
        halves = xp.asarray(self.item_halves)[item]
        halves = xp.where(flip, halves[::-1], halves)
        return task | {
            "kind": xp.where(at, xp.where(is_cup, CUP, PIN), task["kind"]),
            "pin_color": xp.where((at & ~is_cup)[..., None], halves, task["pin_color"]),
            "cup_opaque_lower": xp.where(at & is_cup, flip, task["cup_opaque_lower"]),
            "height": task["height"] + (on_goal & ok),
            "held": task["held"] - (self._one(task, ctx["agent"], item, xp) & ok),
        }, ok

    def remove(self, task, ctx, xp):
        goal = ctx["args"][0]
        height = task["height"][goal]
        ok = height > 0
        top = xp.maximum(height - 1, 0)
        on_goal = xp.arange(len(self.goals)) == goal
        at = on_goal[:, None] & (xp.arange(self.levels) == top)[None, :] & ok
        kind = task["kind"][goal, top]
        colors = task["pin_color"][goal, top]
        item = xp.where(kind == CUP, self.cup, xp.asarray(self.pin_item)[colors[0], colors[1]])
        returned = (xp.arange(len(self.items)) == item) & ok & (kind != EMPTY)
        return task | {
            "kind": xp.where(at, EMPTY, task["kind"]),
            "pin_color": xp.where(at[..., None], 0, task["pin_color"]),
            "cup_opaque_lower": task["cup_opaque_lower"] & ~at,
            "crowded": task["crowded"] & ~at,
            "height": task["height"] - (on_goal & ok),
            "supply": task["supply"] + returned,
        }, ok

    def pick(self, task, ctx, xp):
        return self._take(task, ctx, ctx["args"][0], xp)

    def load(self, task, ctx, xp):
        return self._take(task, ctx, ctx["args"][1], xp)

    def flip_toggle(self, task, ctx, xp):
        at = xp.arange(len(self.quadrants)) == xp.asarray(self.toggle_quadrant)[ctx["args"][0]]
        face = xp.asarray(self.alliance_face)[ctx["alliance"]]
        return task | {
            "toggle_face": xp.where(at, face, task["toggle_face"]),
            "toggle_set": task["toggle_set"] | at,
        }, xp.asarray(True)

    def _take(self, task, ctx, item, xp):
        ok = (task["supply"][item] > 0) & (xp.sum(task["held"][ctx["agent"]]) < ctx["carry"])
        return task | {
            "supply": task["supply"] - ((xp.arange(len(self.items)) == item) & ok),
            "held": task["held"] + (self._one(task, ctx["agent"], item, xp) & ok),
        }, ok

    def _one(self, task, agent, item, xp):
        """A [agent, item] mask that is True at one robot and item."""
        robots, items = task["held"].shape
        return (xp.arange(robots) == agent)[:, None] & (xp.arange(items) == item)[None, :]

    def on_checkpoint(self, task, checkpoint, scores, xp):
        red, blue = scores[checkpoint, 0], scores[checkpoint, 1]
        codes = self.winner_codes
        winner = xp.where(
            red > blue, codes["red"], xp.where(blue > red, codes["blue"], codes["tie"])
        )
        now = (checkpoint == self.autonomous_end) & ~task["winner_known"]
        return task | {
            "autonomous_winner": xp.where(now, winner, task["autonomous_winner"]),
            "winner_known": task["winner_known"] | now,
        }

    def task_state(self, task, robots, checkpoint, xp):
        inside = polygon_distances(robots["xy"], self.midfield, xp) < self.reach
        counts = xp.stack([xp.sum(inside & (robots["alliance"] == k)) for k in self.alliance_slot])
        return {
            "kind": task["kind"],
            "pin_color": task["pin_color"],
            "cup_opaque_lower": task["cup_opaque_lower"],
            "crowded": task["crowded"],
            "toggle_face": task["toggle_face"],
            "toggle_set": task["toggle_set"],
            "robots_in_midfield": counts.astype(xp.int8),
            "checkpoint": xp.asarray(self.checkpoint_codes)[checkpoint],
            "program": xp.asarray(self.spec.code("program", self.program), xp.int8),
            "autonomous_winner": task["autonomous_winner"],
        }
