"""Top-down replay plot for debugging.

replay_svg(env, states) draws the field with its regions and object instances, each robot's path
through the given states (one State per step, as replay_episode(..., trace=True) keeps them),
and the points at each checkpoint. It returns SVG text; no plotting library is needed.
"""

from html import escape
from pathlib import Path

import numpy as np

from spar_sim.env import PlanEnv, State

SCALE = 160  # pixels per meter
MARGIN = 40
LABEL_ROOM = 80  # past the right wall, for labels of things on it
COLORS = ["#c62828", "#1565c0"]  # the first and second alliance


def replay_svg(env: PlanEnv, states: list[State], *, title: str = "") -> str:
    width_m, height_m = env.scene.spec.field.size_m
    w = width_m * SCALE + MARGIN + LABEL_ROOM
    h = height_m * SCALE + 2 * MARGIN + 30

    def px(x, y):
        return MARGIN + (x + width_m / 2) * SCALE, MARGIN + 30 + (height_m / 2 - y) * SCALE

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
        f'font-family="sans-serif" font-size="11">',
        f'<rect width="{w:.0f}" height="{h:.0f}" fill="#fafafa"/>',
        f'<rect x="{MARGIN}" y="{MARGIN + 30}" width="{width_m * SCALE:.0f}" '
        f'height="{height_m * SCALE:.0f}" fill="#eceff1" stroke="#455a64" stroke-width="2"/>',
    ]
    for polygon in env.scene.regions.values():
        points = " ".join("{:.1f},{:.1f}".format(*px(*p)) for p in polygon)
        parts.append(
            f'<polygon points="{points}" fill="none" stroke="#90a4ae" stroke-dasharray="4 3"/>'
        )
    for name, (x, y) in env.scene.instance_positions.items():
        cx, cy = px(x, y)
        parts.append(
            f'<rect x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" fill="#78909c"/>'
        )
        parts.append(
            f'<text x="{cx + 7:.1f}" y="{cy + 4:.1f}" fill="#37474f">{escape(name)}</text>'
        )

    xy = np.array([np.asarray(state.xy) for state in states])  # [step, agent, 2]
    for i, agent in enumerate(env.agents):
        color = COLORS[int(env.alliance[i])]
        path = " ".join("{:.1f},{:.1f}".format(*px(*p)) for p in xy[:, i])
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" '
            'opacity="0.7"/>'
        )
        sx, sy = px(*xy[0, i])
        ex, ey = px(*xy[-1, i])
        parts.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="white" stroke="{color}" '
            'stroke-width="2"/>'
        )
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="7" fill="{color}"/>')
        # Robots that end in the same spot get their labels on separate lines.
        crowd = sum(np.hypot(*(xy[-1, j] - xy[-1, i])) * SCALE < 12 for j in range(i))
        ly = ey - 6 + 13 * crowd
        parts.append(f'<text x="{ex + 9:.1f}" y="{ly:.1f}" fill="{color}">{escape(agent)}</text>')

    scores = np.asarray(states[-1].scores)
    summary = []
    for c, checkpoint in enumerate(env.scene.checkpoints):
        if not np.isnan(scores[c]).any():
            points = "-".join(f"{p:.0f}" for p in scores[c])
            summary.append(f"{checkpoint} {points}")
    line = f"{title}  t = {float(states[-1].t):.1f} s  " + "  ".join(summary)
    parts.append(
        f'<text x="{MARGIN}" y="{MARGIN + 12}" font-size="14">{escape(line.strip())}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def save_replay_svg(
    env: PlanEnv, states: list[State], path: str | Path, *, title: str = ""
) -> None:
    Path(path).write_text(replay_svg(env, states, title=title), encoding="utf-8")
