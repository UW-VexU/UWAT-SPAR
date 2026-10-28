# Override (VEX, 2026 to 2027)

The main test case for SPAR. Rules version `manual-v2.1`:
[game manual v2.1](https://manuals.vex.com/v5/override/v2.1/game/).

This is a plain-language summary. Check the manual before relying on any detail.

- **Alliances.** Red and blue, two robots each. In V5RC an alliance is two teams with one robot
  each and autonomous lasts 15 s. In VEX U it is one team with a 15 in and a 24 in robot, and
  autonomous lasts 30 s (rule VUT4).
- **Goals.** Nine: two red, two blue, four short neutral and one tall neutral. Adding to or taking
  from the opposing alliance's goal is a major violation.
- **Pins and cups.** Pins are the main scoring object. Each pin half is alliance-colored (5 points)
  or yellow (10 points) and scores separately. A cup counts as placed only when nested with a
  placed pin.
- **Toggles.** Four, one per quadrant. Whichever alliance color a Toggle shows owns the yellow pin
  halves in that quadrant.
- **Midfield.** The center region. Each robot ending there scores 8 points, and the alliance with
  more robots there owns the Midfield goal's yellow halves (a tie scores nothing). The last 10 s
  are the endgame, a race for the Midfield.
- **Autonomous bonus.** 12 points to the winner of the autonomous period, 6 each on a tie.

## Files

| File | Contents |
|---|---|
| `scene.yaml` | Field, objects, programs, timing, teacher prompts, objective |
| `scoring.py` | `score(state, xp)`: points per alliance |
| `strategy_code.yaml` | Strategy code fields, distances, weights, coder threshold |
| `landmarks.yaml` | Named field points for calibration |
| `robot_limits.yaml` | Our robots' speed and carry limits |
