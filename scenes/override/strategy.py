"""Override strategy plugin: each alliance's five-field strategy code (strategy_code.yaml).

    auton_route     Goals and Toggles the alliance touched in autonomous, in order: its robots'
                    place and toggle_flip events up to the end of autonomous.
    goal_order      For each robot, the Goals it placed on over the match, in order.
    toggle_share    For each Quadrant's Toggle, in scene order, the share of the match it was set
                    to the alliance's color (seated, not touched, showing that face).
    time_split      For each robot, its share of time near a Toggle, near its alliance's Loaders,
                    near a Goal it can score on, or near an opponent away from those (defense),
                    checked in that order. Time near none of them does not count, and a robot
                    never near any of them gets an even split.
    midfield_entry  For each robot, when it last entered the Midfield, or null if it never did.
Repeats in a row are collapsed in sequences. Robots are listed in agent ID order. The distances
below are first versions, to tune against hand-coded matches.
"""

import json
import math

NEAR_TOGGLE_M = 0.5
NEAR_LOADER_M = 0.6
NEAR_GOAL_M = 0.5
NEAR_OPPONENT_M = 0.7


def extract_code(episode, events, scene):
    program = scene.programs[episode.meta.program]
    match_length = float(program.autonomous_s + program.driver_s)
    autonomous_end = scene.checkpoint_time("autonomous_end", episode.meta.program)
    robots = {
        alliance: sorted(
            a for a, entry in episode.meta.roster.items() if entry.alliance == alliance
        )
        for alliance in scene.alliances
    }
    places = [e for e in events if e["type"] == "place" and e["goal_id"]]
    flips = [e for e in events if e["type"] == "toggle_flip"]
    shares = _toggle_shares(episode, scene, match_length)
    splits = _time_splits(episode, scene, robots, match_length)

    codes = {}
    for alliance, agents in robots.items():
        touched = sorted(
            [(e["t"], e["goal_id"]) for e in places if e["agent_id"] in agents]
            + [(e["t"], e["target_id"]) for e in flips if e["agent_id"] in agents]
        )
        entries = {
            agent: [
                e["t"] for e in events if e["type"] == "enter_midfield" and e["agent_id"] == agent
            ]
            for agent in agents
        }
        codes[alliance] = {
            "auton_route": _collapse([target for t, target in touched if t <= autonomous_end]),
            "goal_order": [
                _collapse([e["goal_id"] for e in places if e["agent_id"] == agent])
                for agent in agents
            ],
            "toggle_share": [round(share, 6) for share in shares[alliance]],
            "time_split": [splits[agent] for agent in agents],
            "midfield_entry": [max(entries[agent]) if entries[agent] else None for agent in agents],
        }
    return codes


def _collapse(items):
    return [item for i, item in enumerate(items) if i == 0 or item != items[i - 1]]


def _toggle_shares(episode, scene, match_length):
    """For each alliance, each Quadrant Toggle's share of the match set to its color."""
    instances = scene.spec.objects["toggle"]["instances"]
    by_quadrant = {spec["quadrant"]: toggle for toggle, spec in instances.items()}
    order = [by_quadrant[q] for q in scene.task_state.dims["quadrant"]]
    rows = (
        []
        if episode.objects is None
        else [r for r in episode.objects.to_pylist() if r["cls"] == "toggle"]
    )
    shares = {alliance: [] for alliance in scene.alliances}
    for toggle in order:
        changes = sorted((r["t"], _set_color(r)) for r in rows if r["object_id"] == toggle)
        held = {alliance: 0.0 for alliance in scene.alliances}
        for k, (t, color) in enumerate(changes):
            end = changes[k + 1][0] if k + 1 < len(changes) else match_length
            span = max(0.0, min(end, match_length) - max(t, 0.0))
            if color in held:
                held[color] += span
        for alliance in scene.alliances:
            shares[alliance].append(held[alliance] / match_length)
    return shares


def _set_color(row):
    attrs = json.loads(row["attrs"]) if row["attrs"] else {}
    if not attrs.get("seated", True) or attrs.get("touched", False):
        return "yellow"
    return attrs.get("face", "yellow")


def _time_splits(episode, scene, robots, match_length):
    """Each robot's shares of time at Toggles, Loaders, scoring Goals and defense."""
    objects = scene.spec.objects
    toggles = [tuple(s["xy"]) for s in objects["toggle"]["instances"].values()]
    goals = objects["goal"]["instances"]
    alliance_of = {agent: alliance for alliance, agents in robots.items() for agent in agents}
    counts = {agent: [0, 0, 0, 0] for agent in alliance_of}  # scoring, loading, defense, toggles
    rows = (
        []
        if episode.tracks is None
        else episode.tracks.select(["t", "agent_id", "x", "y"]).to_pylist()
    )
    by_time = {}
    for row in rows:
        if row["t"] <= match_length and row["agent_id"] in alliance_of:
            by_time.setdefault(row["t"], []).append(row)
    for robots_now in by_time.values():
        for row in robots_now:
            agent, xy = row["agent_id"], (row["x"], row["y"])
            alliance = alliance_of[agent]
            loaders = [
                tuple(s["xy"])
                for s in objects["loader"]["instances"].values()
                if s["alliance"] == alliance
            ]
            scoring = [
                tuple(s["xy"])
                for s in goals.values()
                if s["kind"] in ("neutral_short", "neutral_tall", f"alliance_{alliance}")
            ]
            opponents = [
                (r["x"], r["y"]) for r in robots_now if alliance_of[r["agent_id"]] != alliance
            ]
            if _near(xy, toggles, NEAR_TOGGLE_M):
                counts[agent][3] += 1
            elif _near(xy, loaders, NEAR_LOADER_M):
                counts[agent][1] += 1
            elif _near(xy, scoring, NEAR_GOAL_M):
                counts[agent][0] += 1
            elif _near(xy, opponents, NEAR_OPPONENT_M):
                counts[agent][2] += 1
    splits = {}
    for agent, parts in counts.items():
        total = sum(parts)
        splits[agent] = [p / total for p in parts] if total else [0.25, 0.25, 0.25, 0.25]
    return splits


def _near(xy, points, distance):
    return any(math.hypot(xy[0] - p[0], xy[1] - p[1]) <= distance for p in points)
