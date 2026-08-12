"""Where the weights live.

`setup.sh` downloads into the repository directory, so the defaults resolve there
too and a fresh clone runs without arguments. Each can be overridden by an
environment variable, or by the matching command-line flag.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _p(env, *default):
    return str(Path(os.environ.get(env) or ROOT.joinpath(*default)))


ASSETS = _p("LIVEWAN_ASSETS", "assets")               # the LiveWan HF repo
BASE_DIR = _p("LIVEWAN_BASE_DIR", "wan21_13b")        # VAE, umt5, base transformer
WAN_REPO = _p("LIVEWAN_WAN_REPO", "wan21_repo")       # the Wan2.1 reference code
WORLDS_DIR = _p("LIVEWAN_WORLDS_DIR", "generated_worlds")
WEIGHTS = str(Path(ASSETS) / "checkpoints/t14b_b64/latest.pt")
