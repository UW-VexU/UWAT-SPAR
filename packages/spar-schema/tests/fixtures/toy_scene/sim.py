"""Simulator plugin for the toy test scene: one point per ball dropped in a goal.

The task arrays are points[alliance], supply[item] (balls not held or placed) and held[agent,
item].
"""

import numpy as np


def infer_action(event, obj, xy, scene):
    if event["type"] == "pick":
        return "pick", {"item": "ball"}
    if event["type"] == "place":
        return "place", {"goal": event["goal_id"]}
    if event["type"] == "enter_zone":
        return "park", {}
    return None


def rules(scene, program):
    return Rules(scene)


class Rules:
    def __init__(self, scene):
        self.alliances = scene.alliances
        self.balls = scene.spec.objects["ball"]["count"]
        self.effects = {"pick": self.pick, "place": self.place}

    def encode(self, start, agents):
        points = np.zeros(2, np.int16)
        held = np.zeros((len(agents), 1), np.int8)
        for obj in start.objects:
            if obj["status"] == "placed" and obj.get("goal_id") in self.alliances:
                points[self.alliances.index(obj["goal_id"])] += 1
            elif obj["status"] == "held" and obj.get("holder") in agents:
                held[agents.index(obj["holder"]), 0] += 1
        supply = np.array([self.balls - points.sum() - held.sum()], np.int16)
        return {"points": points, "supply": supply, "held": held}

    def pick(self, task, ctx, xp):
        mine = xp.arange(task["held"].shape[0]) == ctx["agent"]
        ok = (task["supply"][0] > 0) & (xp.sum(task["held"][ctx["agent"]]) < ctx["carry"])
        return {
            "points": task["points"],
            "supply": task["supply"] - ok,
            "held": task["held"] + (mine & ok)[:, None],
        }, ok

    def place(self, task, ctx, xp):
        mine = xp.arange(task["held"].shape[0]) == ctx["agent"]
        goal = xp.arange(2) == ctx["args"][0]
        ok = task["held"][ctx["agent"], 0] > 0
        return {
            "points": task["points"] + (goal & ok),
            "supply": task["supply"],
            "held": task["held"] - (mine & ok)[:, None],
        }, ok

    def on_checkpoint(self, task, checkpoint, scores, xp):
        return task

    def task_state(self, task, robots, checkpoint, xp):
        return {
            "points": task["points"],
            "holder": xp.zeros(3, xp.int8),
            "over": xp.asarray(checkpoint == 1),
        }
