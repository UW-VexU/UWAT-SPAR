"""State plugin for the toy test scene: one point per ball Placed on an alliance's goal."""

import numpy as np


def build_state(snapshot, scene):
    spec = scene.task_state
    state = spec.empty(np)
    alliances = spec.dims["alliance"]
    for obj in snapshot.objects:
        if obj["cls"] == "ball" and obj["status"] == "placed" and obj["goal_id"] in alliances:
            state["points"][alliances.index(obj["goal_id"])] += 1
    state["over"] = np.asarray(snapshot.checkpoint == "match_end")
    return state
