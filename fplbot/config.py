from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent

# Profiles let one checkout drive several FPL accounts. A profile selects:
#   - the .env file that is loaded (and that OIDCAuth writes the rotated refresh token to)
#   - a per-profile state dir for the decisions DB (idempotency records)
# Public-data snapshots (DATA_DIR/raw, player_snapshots) stay shared.
# Selection: --profile NAME (CLI) or FPLBOT_PROFILE -> ROOT/.env.<NAME>;
#            FPLBOT_ENV_FILE overrides the env file path explicitly.
PROFILE: str | None = None
ENV_FILE: Path = ROOT / ".env"
DATA_DIR: Path = ROOT / "data"
STATE_DIR: Path = DATA_DIR

_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_loaded_keys: set[str] = set()  # keys we injected from a .env (so a profile switch can drop them)


def resolve_env_file(profile: str | None = None, environ: dict | None = None) -> Path:
    """Which .env to use: FPLBOT_ENV_FILE > ROOT/.env.<profile> > ROOT/.env."""
    environ = os.environ if environ is None else environ
    explicit = environ.get("FPLBOT_ENV_FILE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if profile:
        if not _PROFILE_RE.match(profile):
            raise ValueError(f"Invalid profile name {profile!r} (use letters, digits, _ or -)")
        return ROOT / f".env.{profile}"
    return ROOT / ".env"


def activate_profile(profile: str | None = None) -> Path:
    """(Re)load env + paths for a profile. Called at import with FPLBOT_PROFILE and again by the
    CLI's --profile callback. Variables injected by a previously loaded .env are removed first so
    one account's credentials can never leak into another profile; real environment variables
    (e.g. CI secrets) still win over .env values."""
    global PROFILE, ENV_FILE, DATA_DIR, STATE_DIR
    profile = profile or os.environ.get("FPLBOT_PROFILE") or None
    env_file = resolve_env_file(profile)
    for k in _loaded_keys:
        os.environ.pop(k, None)
    _loaded_keys.clear()
    if env_file.exists():
        for k, v in dotenv_values(env_file).items():
            if v is not None and k not in os.environ:
                os.environ[k] = v
                _loaded_keys.add(k)
    PROFILE, ENV_FILE = profile, env_file
    DATA_DIR = Path(os.environ.get("FPLBOT_DATA_DIR", ROOT / "data"))
    STATE_DIR = DATA_DIR / profile if profile else DATA_DIR
    return env_file


activate_profile()

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    def __getattr__(self, item):
        try:
            return self.raw[item]
        except KeyError as e:
            raise AttributeError(item) from e

    @property
    def entry_id(self) -> int | None:
        v = os.environ.get("FPL_ENTRY_ID")
        return int(v) if v else None


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return Config(yaml.safe_load(f))
