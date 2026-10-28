"""Override scoring plugin (spec section 5.3).

Must encode: when a pin or cup counts as Placed (nesting rules), alliance and yellow pin-half
values, Toggle ownership of yellow halves per quadrant, Midfield goal ownership by robot count
(a tie scores nothing), robots ending in the Midfield, and the autonomous bonus. Unit tests must
cover each rule with hand-built states.
"""


def score(state, xp) -> dict[str, float]:
    """Points per alliance for one state, e.g. {"red": 57.0, "blue": 40.0}.

    xp is the array module (numpy or jax.numpy). Write the rules with
    array operations only, so the same code runs in spar-state (NumPy)
    and inside the JAX simulator.
    """
    raise NotImplementedError
