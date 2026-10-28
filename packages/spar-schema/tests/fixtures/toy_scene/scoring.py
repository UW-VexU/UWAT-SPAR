"""Scoring plugin for the toy test scene: the state already holds each alliance's points."""


def score(state, xp):
    points = xp.asarray(state["points"])
    return {"red": points[0], "blue": points[1]}
