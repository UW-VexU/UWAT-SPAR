"""The `spar` command. Each package defines its own Typer app in cli.py; this only mounts them."""

import typer

from spar_data.cli import app as data_app
from spar_eval.cli import app as eval_app
from spar_gt.cli import app as gt_app
from spar_ingest.cli import app as ingest_app
from spar_learn.cli import app as learn_app
from spar_perception.cli import app as perception_app
from spar_sim.cli import app as sim_app
from spar_state.cli import app as state_app

app = typer.Typer(help="SPAR: Synthetic Play from Archived Replays.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(perception_app, name="perception")
app.add_typer(state_app, name="state")
app.add_typer(data_app, name="data")
app.add_typer(gt_app, name="gt")
app.add_typer(sim_app, name="sim")
app.add_typer(learn_app, name="learn")
app.add_typer(eval_app, name="eval")
