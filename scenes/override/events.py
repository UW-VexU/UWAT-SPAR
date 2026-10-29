"""Override events plugin: Toggle flips.

extra_events() adds a toggle_flip event each time a Toggle's face changes, with target_id the
Toggle's ID, credited to the robot nearest the Toggle's wall within reach. The generic rules in
spar-state already give pick, place and remove events, and enter_midfield and leave_midfield.
"""

import json

REACH_M = 0.6


def extra_events(episode, scene, nearest):
    if episode.objects is None:
        return []
    toggles = scene.spec.objects["toggle"]["instances"]
    rows = sorted(
        (row for row in episode.objects.to_pylist() if row["cls"] == "toggle"),
        key=lambda row: (row["object_id"], row["t"]),
    )
    found, face_before = [], {}
    for row in rows:
        face = (json.loads(row["attrs"]) if row["attrs"] else {}).get("face", "yellow")
        toggle = row["object_id"]
        if toggle in face_before and face != face_before[toggle]:
            xy = tuple(toggles[toggle]["xy"])
            found.append(
                {
                    "t": float(row["t"]),
                    "agent_id": nearest(row["t"], xy, REACH_M),
                    "type": "toggle_flip",
                    "object_id": None,
                    "goal_id": None,
                    "target_id": toggle,
                    "conf": row["conf"],
                    "source": "rule",
                }
            )
        face_before[toggle] = face
    return found
