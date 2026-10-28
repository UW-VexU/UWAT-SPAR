# Override (VEX, 2026 to 2027)

The main test case for SPAR. Rules version `manual-v2.1`:
[game manual v2.1](https://manuals.vex.com/v5/override/v2.1/game/), released October 8, 2026.
Rules, timing and points were checked against v2.1. Field geometry comes from the Appendix A
drawings in the v1.1 PDF, because v2.1 is only published online; the v2.0 and v2.1 changelogs list
no field changes.

This is a plain-language summary. Check the manual before relying on any detail.

## Rules that decide the score

- **Alliances.** Red and blue, two Robots each. In V5RC an alliance is two Teams with one Robot
  each; autonomous is 15 s and driver control 1:45. In VEX U it is one Team with a 15 in and a
  24 in Robot; autonomous is 30 s (VUT4) and driver control 90 s (VUT5).
- **Placed (SC2).** A stack is built up from the Goal: a Pin nested with a Goal, then a Cup on it,
  then a Pin in the Cup, and so on. A Goal, or either half of a Cup, may hold only one Pin Half;
  a Pin in a crowded socket is not Placed, and neither is anything that rests on it.
- **Scored (SC3).** Each half of a Placed Pin scores if it is visible. A half inside the Goal, or
  inside the opaque (gray) half of a Cup, is not visible. Cups have one gray half and one clear
  half, so which way up a Cup sits decides which neighboring Pin Half it hides.
- **Points.** A red or blue Pin Half scores 5 for its alliance. A yellow Pin Half scores 10 for
  the alliance that Owns it.
- **Toggles (SC4, SC5).** Each Quadrant has one Toggle. Yellow halves on that Quadrant's two Goals
  belong to the alliance whose color the Toggle is set to. A Toggle that is yellow, not fully
  seated, or touched by a Robot gives them to nobody.
- **Midfield (SC5, SC6).** Yellow halves on the Midfield Goal belong to the alliance with more
  Robots at least partially in the Midfield at the end, and to nobody on a tie. Each of those
  Robots also scores 8.
- **Autonomous Bonus (SC7).** 12 points to the alliance ahead when autonomous ends, 6 each on a
  tie. V5RC leaves the Midfield out of that count; VEX U includes it (VUG5). A Violation during
  autonomous gives the bonus to the other alliance, or to nobody if both commit one.
- **Not points.** The Autonomous Win Point (SC8, VUG6) affects rankings only, so `scoring.py`
  does not compute it. Violations never remove points (SC1e).

## Field map

Seen from the audience: red Alliance Station on the left, blue on the right. The Autonomous Line
runs from the far red corner to the near blue corner, with the red Quadrants on its near-left side.

| Quadrant (Tournament Manager label) | Position | Toggle | Goals |
|---|---|---|---|
| `red_1` | left | `T_red_1`, left wall | `R1` neutral, `R2` red |
| `red_2` | near (audience side) | `T_red_2`, near wall | `R3` red, `R4` neutral |
| `blue_1` | right | `T_blue_1`, right wall | `B1` neutral, `B2` blue |
| `blue_2` | far | `T_blue_2`, far wall | `B3` blue, `B4` neutral |
| Midfield | center | none | `M`, the tall neutral Goal |

Goal IDs are the side of the Autonomous Line plus the AprilTag number printed on the Goal, so
the camera can read them (the Midfield Goal's tag is 0). The layout is symmetric under a half
turn, which swaps red and blue.

## Files

| File | Contents |
|---|---|
| `scene.yaml` | Field, objects with positions, programs, timing, teacher prompts, objective, and the `state` arrays `scoring.py` reads |
| `scoring.py` | `score(state, xp)`: points per alliance, in NumPy or JAX |
| `strategy_code.yaml` | Strategy code fields, distances, weights, coder threshold |
| `landmarks.yaml` | Named field points for calibration |
| `robot_limits.yaml` | Our robots' speed and carry limits |
| `tests/test_scoring.py` | Tests for each scoring rule, including the manual's worked example |

## Still open

| Value | Waiting on |
|---|---|
| `field.frame.x_axis` | Agreement with the physics repo. Coordinates here assume x toward the blue Alliance Station and y away from the audience |
| `robot_limits.yaml` | Our robots' measured speed and carry limits |
| `strategy_code.yaml` `missing_penalty` | The coder study |
| Teacher prompts | Tuning on real frames (task E5) |
