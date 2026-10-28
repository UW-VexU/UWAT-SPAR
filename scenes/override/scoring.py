"""Override scoring plugin (spec section 5.3), following game manual v2.1.

score(state, xp) returns the points per alliance for one TaskState, laid out as in the `state`
section of scene.yaml. It uses array operations only, so the same code runs with NumPy in
spar-state and with jax.numpy inside the simulator, including under jax.jit and jax.vmap.

Rules, by manual rule number:
  SC2   A Pin is Placed if nested with a Goal, or with a Cup that is nested with another Placed
        Pin, and every Goal or Cup half it is nested with holds at most one Pin Half. A Cup is
        Placed if nested with a Placed Pin. So a scoring stack is built up from the Goal.
  SC3   Each half of a Placed Pin is Scored if visible, meaning not nested inside a Goal or the
        opaque half of a Cup. Red halves score for red, blue halves for blue, and yellow halves
        for the Alliance that Owns them.
  SC4   A Toggle is set only when fully seated and not touched by a Robot. Otherwise it is yellow.
  SC5   Yellow halves on a Quadrant's Goals are Owned by the Alliance its Toggle is set to. Yellow
        halves on the Midfield Goal are Owned by the Alliance with more Robots in the Midfield,
        and by nobody on a tie.
  SC6   Each Robot at least partially within the Midfield scores.
  SC7   At autonomous_end, V5RC leaves out the Midfield: Robots there and ownership of its yellow
        halves. VEX U counts both (VUG5). At match_end the Autonomous Bonus goes to the autonomous
        winner, or half to each Alliance on a tie.

Violations do not change points (SC1e). Their effect on the Autonomous Bonus (SC7c, SC7d) comes
in through state["autonomous_winner"].
"""

# Codes, matching the `values` lists in scene.yaml (a test checks that they agree).
EMPTY, PIN, CUP = 0, 1, 2
YELLOW, RED, BLUE = 0, 1, 2
AUTONOMOUS_END, MATCH_END = 0, 1
V5RC, VEXU = 0, 1
WINNER_NONE, WINNER_RED, WINNER_BLUE, WINNER_TIE = 0, 1, 2, 3

# Which owner each Goal's yellow halves follow, in the scene.yaml goal order R1-R4, B1-B4, M:
# the index of the Goal's Quadrant (red_1, red_2, blue_1, blue_2), or 4 for the Midfield.
GOAL_OWNER = (0, 0, 1, 1, 2, 2, 3, 3, 4)

# Points, matching objective.points in scene.yaml.
AUTONOMOUS_BONUS = 12
AUTONOMOUS_TIE_BONUS = 6
ALLIANCE_PIN_HALF = 5
YELLOW_PIN_HALF = 10
ROBOT_IN_MIDFIELD = 8


def score(state, xp):
    """Points per alliance for one state, as 0-d float32 arrays: {"red": ..., "blue": ...}.

    xp is the array module (numpy or jax.numpy).
    """
    kind = xp.asarray(state["kind"])  # [goal, level], bottom first
    is_pin = kind == PIN
    is_cup = kind == CUP
    in_goal = xp.arange(kind.shape[1]) == 0

    def below(a):  # the value one level down, or zero at level 0
        return xp.concatenate([xp.zeros_like(a[:, :1]), a[:, :-1]], axis=1)

    def above(a):  # the value one level up, or zero at the top
        return xp.concatenate([a[:, 1:], xp.zeros_like(a[:, :1])], axis=1)

    # SC2: a Pin must not share a socket it is nested in, either the one it sits in or the
    # bottom half of a Cup on top of it. Each level must then rest on the right thing, all the
    # way down to the Goal.
    crowded = xp.asarray(state["crowded"])
    pin_ok = is_pin & ~crowded & ~(above(is_cup) & above(crowded))
    rests_on = (pin_ok & below(is_cup)) | (is_cup & below(is_pin))
    placed = xp.cumprod(xp.where(in_goal, pin_ok, rests_on).astype(xp.int8), axis=1) == 1

    # SC3: a Pin Half is hidden inside the Goal, or inside the opaque half of a Cup.
    opaque_lower = xp.asarray(state["cup_opaque_lower"])
    lower_hidden = in_goal | below(is_cup & ~opaque_lower)
    upper_hidden = above(is_cup & opaque_lower)
    hidden = xp.stack([lower_hidden, upper_hidden], axis=-1)
    scored = (placed & is_pin)[..., None] & ~hidden  # [goal, level, half]
    color = xp.asarray(state["pin_color"])
    red_halves = xp.sum(scored & (color == RED))
    blue_halves = xp.sum(scored & (color == BLUE))
    yellow_halves = xp.sum(scored & (color == YELLOW), axis=(1, 2))  # per Goal

    # SC4, SC5a: each Quadrant's yellow halves follow its Toggle, if the Toggle is set.
    face = xp.where(xp.asarray(state["toggle_set"]), xp.asarray(state["toggle_face"]), YELLOW)

    # SC5b, SC6, SC7a, VUG5: the Midfield counts at match end, and at autonomous end in VEX U.
    checkpoint = xp.asarray(state["checkpoint"])
    midfield_counts = (checkpoint == MATCH_END) | (xp.asarray(state["program"]) == VEXU)
    robots = xp.asarray(state["robots_in_midfield"]).astype(xp.int32) * midfield_counts
    midfield_red = robots[0] > robots[1]
    midfield_blue = robots[1] > robots[0]

    owner = xp.asarray(GOAL_OWNER)
    owned_red = xp.concatenate([face == RED, midfield_red[None]])[owner]
    owned_blue = xp.concatenate([face == BLUE, midfield_blue[None]])[owner]

    red = ALLIANCE_PIN_HALF * red_halves + YELLOW_PIN_HALF * xp.sum(yellow_halves * owned_red)
    blue = ALLIANCE_PIN_HALF * blue_halves + YELLOW_PIN_HALF * xp.sum(yellow_halves * owned_blue)
    red = red + ROBOT_IN_MIDFIELD * robots[0]
    blue = blue + ROBOT_IN_MIDFIELD * robots[1]

    # SC7: the Autonomous Bonus, part of the final score.
    winner = xp.asarray(state["autonomous_winner"])
    at_end = checkpoint == MATCH_END
    tie_bonus = AUTONOMOUS_TIE_BONUS * (winner == WINNER_TIE)
    red = red + at_end * (AUTONOMOUS_BONUS * (winner == WINNER_RED) + tie_bonus)
    blue = blue + at_end * (AUTONOMOUS_BONUS * (winner == WINNER_BLUE) + tie_bonus)
    return {"red": xp.asarray(red, dtype=xp.float32), "blue": xp.asarray(blue, dtype=xp.float32)}
