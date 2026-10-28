"""start_states.parquet round trip."""

import pyarrow.parquet as pq

from spar_schema import (
    SCHEMA_VERSION,
    RobotState,
    StartState,
    read_start_states,
    write_start_states,
)


def test_round_trip(tmp_path, scene):
    # Values exactly representable in float32, so they survive the round trip unchanged.
    states = [
        StartState(
            episode_id="toy_E1_Q1",
            t=12.5,
            time_left_s=47.5,
            robots=[
                RobotState(
                    agent_id="red_1", x=-0.5, y=0.25, heading=1.5, vx=0.25, vy=0.0, omega=0.0
                ),
                RobotState(
                    agent_id="blue_1", x=0.5, y=0.0, heading=-1.5, vx=0.0, vy=0.0, omega=0.5
                ),
            ],
            objects=[{"object_id": "ball_1", "cls": "ball", "status": "free", "x": 0.0, "y": 0.0}],
        ),
        StartState(episode_id="toy_E1_Q1", t=30.0, time_left_s=30.0, robots=[]),
    ]
    path = tmp_path / "start_states.parquet"
    write_start_states(path, states, scene, git_commit="abc123")
    assert read_start_states(path) == states
    metadata = pq.read_schema(path).metadata
    assert metadata[b"spar.schema_version"] == SCHEMA_VERSION.encode()
    assert metadata[b"spar.scene"] == b"toy"
    assert metadata[b"spar.git_commit"] == b"abc123"
