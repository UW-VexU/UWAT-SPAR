"""Override state plugin: builds the TaskState declared in scene.yaml from one Snapshot.

What perception has to record in objects.parquet for this to work:
  - Objects on a Goal: goal_id, and stack_index, the level counting up from the Goal (0 sits in it).
  - Pin attrs: {"halves": [lower, upper]}, each "red", "blue" or "yellow", as the Pin stands.
  - Cup attrs: {"gray": "up" | "down"}, which way its opaque half faces.
  - Toggle rows: object_id is the Toggle's ID from scene.yaml, and attrs are
    {"face": "red" | "blue" | "yellow", "seated": bool, "touched": bool}. A Toggle without a row
    counts as yellow.
Two objects at the same Goal and level mean a crowded socket (SC2).

A Robot counts as in the Midfield (SC6) when the Midfield is within its reach of its center: its
radius_m if the Snapshot has one, otherwise half the smallest legal starting size for the program.
That default only counts Robots clearly inside, until perception measures each Robot's footprint.
"""

import numpy as np

from spar_schema.frames import inches, polygon_distance


def build_state(snapshot, scene):
    if snapshot.checkpoint is None:
        raise ValueError("Override states are scored at a checkpoint: autonomous_end or match_end")
    spec = scene.task_state
    state = spec.empty(np)
    levels = spec.size("level")
    toggles = scene.spec.objects["toggle"]["instances"]
    taken = set()

    for obj in snapshot.objects:
        attrs = obj["attrs"]
        if obj["cls"] in ("pin", "cup") and obj.get("goal_id") is not None:
            g, level = spec.index("goal", obj["goal_id"]), obj["stack_index"]
            if not 0 <= level < levels:
                raise ValueError(
                    f"{obj['object_id']} is at level {level}; the state holds {levels}"
                )
            if (g, level) in taken:
                state["crowded"][g, level] = True
                continue
            taken.add((g, level))
            state["kind"][g, level] = spec.code("kind", obj["cls"])
            if obj["cls"] == "pin":
                state["pin_color"][g, level] = [spec.code("pin_color", c) for c in attrs["halves"]]
            else:
                state["cup_opaque_lower"][g, level] = attrs["gray"] == "down"
        elif obj["cls"] == "toggle":
            q = spec.index("quadrant", toggles[obj["object_id"]]["quadrant"])
            state["toggle_face"][q] = spec.code("toggle_face", attrs.get("face", "yellow"))
            state["toggle_set"][q] = attrs.get("seated", True) and not attrs.get("touched", False)

    midfield = scene.spec.field.regions["midfield"]
    default_reach = inches(min(scene.programs[snapshot.program].robot_sizes_in)) / 2
    for robot in snapshot.robots:
        reach = robot.get("radius_m") or default_reach
        if polygon_distance(robot["x"], robot["y"], midfield) < reach:
            state["robots_in_midfield"][spec.index("alliance", robot["alliance"])] += 1

    for field, label in [
        ("checkpoint", snapshot.checkpoint),
        ("program", snapshot.program),
        ("autonomous_winner", snapshot.autonomous_winner),
    ]:
        state[field] = np.asarray(spec.code(field, label), dtype=np.int8)
    return state
